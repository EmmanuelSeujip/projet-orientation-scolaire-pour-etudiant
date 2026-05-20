import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler

# Seuil : nombre minimum d'exemples vus par le SGD avant de préférer sa prédiction
SGD_TRUST_THRESHOLD = 50

# ── Colonnes et types ─────────────────────────────────────────────────────────

CAT_COLS = [
    'gender_F', 'gender_M',
    'disability_N', 'disability_Y',
    'highest_education',
    'final_result',
]

# Groupes OHE mutuellement exclusifs : une seule colonne du groupe peut valoir 1
OHE_GROUPS = [
    ['gender_F', 'gender_M'],
    ['disability_N', 'disability_Y'],
]

ORDINAL_MAPS = {
    'final_result': {0: 'Withdrawn', 1: 'Fail', 2: 'Pass', 3: 'Distinction'},
    'highest_education': {
        0: 'No Formal quals',
        1: 'Lower Than A Level',
        2: 'A Level or Equivalent',
        3: 'HE Qualification',
        4: 'Post Graduate Qualification',
    },
    'gender_F':      {0: 'Non', 1: 'Oui'},
    'gender_M':      {0: 'Non', 1: 'Oui'},
    'disability_N':  {0: 'Non', 1: 'Oui'},
    'disability_Y':  {0: 'Non', 1: 'Oui'},
}

# Colonnes toujours fournies par le formulaire → jamais prédites, toujours features
ANCHOR_COLS = [
    'gender_F', 'gender_M',
    'disability_N', 'disability_Y',
    'highest_education',
]

class StudentPredictorWithSGD:
    """
    Prédicteur multi-colonnes à deux couches.

    Couche RF  : base stable entraînée une fois sur l'historique.
    Couche SGD : adaptative, mise à jour à chaque formulaire vérifié.
                 Les ANCHOR_COLS sont toujours dans son vecteur d'entrée.
    """

    def __init__(self, n_estimators: int = 50, max_depth: int = 15,
                 sgd_trust_threshold: int = SGD_TRUST_THRESHOLD):
        self.n_estimators        = n_estimators
        self.max_depth           = max_depth
        self.sgd_trust_threshold = sgd_trust_threshold

        self.rf_models_          = {}
        self.rf_metrics_         = {}
        self.sgd_models_         = {}
        self.sgd_samples_seen_   = {}
        self.sgd_features_       = {}
        self.sgd_scalers_       = {}   # {col: StandardScaler}

        self.all_cols_    = []
        self.target_cols_ = []
        self._fitted      = False
        self._sgd_ready   = False


    # ───────────────────────────────────────────────────────────────────────────
    # Couche 1 : entraînement Random Forest
    # ───────────────────────────────────────────────────────────────────────────

    def fit(self, df_encoded: pd.DataFrame, verbose: bool = True) -> "StudentPredictor":
        self.all_cols_    = df_encoded.columns.tolist()
        self.target_cols_ = [c for c in self.all_cols_ if c not in ANCHOR_COLS]
        self.rf_models_   = {}
        self.rf_metrics_  = {}

        if verbose:
            print("── Entraînement Random Forest ──────────────────────────────")

        for target in self.target_cols_:
            features = [c for c in self.all_cols_ if c != target]
            X = df_encoded[features].values
            y = df_encoded[target].values

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if target in CAT_COLS:
                model  = RandomForestClassifier(
                    n_estimators=self.n_estimators, max_depth=self.max_depth,
                    n_jobs=-1, random_state=42
                )
                model.fit(X_tr, y_tr)
                score  = accuracy_score(y_te, model.predict(X_te))
                metric = "accuracy"
            else:
                model  = RandomForestRegressor(
                    n_estimators=self.n_estimators, max_depth=self.max_depth,
                    n_jobs=-1, random_state=42
                )
                model.fit(X_tr, y_tr)
                score  = r2_score(y_te, model.predict(X_te))
                metric = "r2"

            self.rf_models_[target]  = model
            self.rf_metrics_[target] = {"metric": metric, "score": round(float(score), 4)}

            if verbose:
                icon = "✓" if score >= 0.5 else "⚠"
                print(f"  {icon} {target:<32} {metric}={score:.4f}")

        self._fitted = True
        return self


    # ───────────────────────────────────────────────────────────────────────────
    # Couche 2 : initialisation SGD
    # ───────────────────────────────────────────────────────────────────────────

    def init_sgd(self, df_encoded: pd.DataFrame,
                 adapt_ratio: float = 0.2,
                 verbose: bool = True) -> "StudentPredictor":
        """
        Initialise un SGD par colonne cible avec le jeu d'adaptation.

        Vecteur d'entrée SGD pour la cible T :
            [ANCHOR_COLS vrais]  +  [prédictions RF des autres cibles ≠ T]

        Les ancres sont TOUJOURS les vraies valeurs → jamais remplacées
        par une prédiction RF même si d'autres colonnes sont inconnues.
        """
        self._check_fitted()

        _, df_adapt = train_test_split(df_encoded, test_size=adapt_ratio, random_state=42)

        self.sgd_models_       = {}
        self.sgd_samples_seen_ = {}
        self.sgd_features_     = {}

        if verbose:
            print("\n── Initialisation couche SGD ────────────────────────────────")

        for target in self.target_cols_:
            sgd_features = ANCHOR_COLS + [c for c in self.target_cols_ if c != target]
            self.sgd_features_[target] = sgd_features

            X_sgd = self._build_sgd_X(df_adapt, target, sgd_features)
            y_sgd = df_adapt[target].values

            # Normaliser les features (crucial pour la stabilité du SGD)
            scaler = StandardScaler()
            X_sgd_scaled = scaler.fit_transform(X_sgd)
            self.sgd_scalers_[target] = scaler

            if target in CAT_COLS:
                from sklearn.utils.class_weight import compute_class_weight
                classes   = np.unique(df_encoded[target].values)
                cw_vals   = compute_class_weight('balanced', classes=classes,
                                                 y=df_encoded[target].values)
                cw_dict   = dict(zip(classes.astype(int), cw_vals))
                model     = SGDClassifier(loss='log_loss', max_iter=1,
                                          warm_start=True, random_state=42,
                                          class_weight=cw_dict)
                model.partial_fit(X_sgd_scaled, y_sgd, classes=classes)
                score  = accuracy_score(y_sgd, model.predict(X_sgd_scaled))
                metric = "accuracy"
            else:
                model  = SGDRegressor(loss='squared_error', max_iter=1,
                                      warm_start=True, random_state=42)
                model.partial_fit(X_sgd_scaled, y_sgd)
                score  = r2_score(y_sgd, model.predict(X_sgd_scaled))
                metric = "r2"

            self.sgd_models_[target]       = model
            self.sgd_samples_seen_[target] = len(y_sgd)

            if verbose:
                n = len(y_sgd)
                print(f"  ✓ {target:<32} {metric}={score:.4f}  ({n} exemples init)")

        self._sgd_ready = True
        return self


    # ───────────────────────────────────────────────────────────────────────────
    # Prédiction combinée RF + SGD
    # ───────────────────────────────────────────────────────────────────────────

    def predict(self, known: dict,
                decode_labels: bool = True,
                confidence: bool = True) -> pd.DataFrame:
        """
        Prédit les colonnes manquantes.

        Règle de bascule RF → SGD :
          SGD actif seulement si examples vus >= sgd_trust_threshold

        known doit TOUJOURS contenir toutes les ANCHOR_COLS.
        """
        self._check_fitted()

        missing_anchors = [c for c in ANCHOR_COLS if c not in known]
        if missing_anchors:
            raise ValueError(
                f"Colonnes ancres manquantes : {missing_anchors}\n"
                "Ces colonnes doivent toujours être fournies par le formulaire."
            )

        known = dict(known)
        for group in OHE_GROUPS:
            known_in_group = {c: known[c] for c in group if c in known}
            if known_in_group:
                provided_1 = [c for c, v in known_in_group.items() if v == 1]
                for col in group:
                    if col not in known:
                        known[col] = 0 if provided_1 else None

        to_predict = [c for c in self.target_cols_
                      if c not in known or known.get(c) is None]

        if not to_predict:
            return pd.DataFrame({"message": ["Toutes les colonnes cibles sont déjà connues."]})

        rows = []
        for target in to_predict:
            use_sgd = (
                self._sgd_ready
                and self.sgd_samples_seen_.get(target, 0) >= self.sgd_trust_threshold
            )

            if use_sgd:
                pred_val, conf, source = self._sgd_predict(target, known)
            else:
                pred_val, conf, source = self._rf_predict(target, known)

            row = {"colonne": target, "prediction": pred_val, "source": source}

            if decode_labels and target in ORDINAL_MAPS:
                row["label"] = ORDINAL_MAPS[target].get(int(round(pred_val)), str(pred_val))
            if confidence and target in CAT_COLS:
                row["confiance"] = f"{conf*100:.1f}%" if conf is not None else "N/A"

            rows.append(row)

        return pd.DataFrame(rows).set_index("colonne")


    # ───────────────────────────────────────────────────────────────────────────
    # Adaptation : partial_fit() sur un formulaire vérifié
    # ───────────────────────────────────────────────────────────────────────────

    def adapt(self, verified_row: dict, verbose: bool = False) -> None:
        """
        Met à jour la couche SGD avec un formulaire complet et vérifié.

        - Les ANCHOR_COLS garantissent un vecteur d'entrée stable.
        - Seules les colonnes cibles présentes dans verified_row sont mises à jour.
        - Appelle partial_fit() → aucun réentraînement complet.

        Exemple :
            pred.adapt({
                'gender_F': 1, 'gender_M': 0,
                'disability_N': 1, 'disability_Y': 0,
                'highest_education': 2,
                'final_result': 2,
                'taux_participation': 0.78,
                'note_moyenne': 68.0,
                ...
            })
        """
        self._check_fitted()
        if not self._sgd_ready:
            raise RuntimeError("SGD non initialisé. Appelle init_sgd() d'abord.")

        missing_anchors = [c for c in ANCHOR_COLS if c not in verified_row]
        if missing_anchors:
            raise ValueError(f"Ancres manquantes dans le formulaire : {missing_anchors}")

        updated = []
        for target in self.target_cols_:
            if target not in verified_row:
                continue

            y_val     = np.array([verified_row[target]])
            sgd_feats = self.sgd_features_[target]
            x_vec     = np.array([
                verified_row.get(f, 0.0) for f in sgd_feats
            ]).reshape(1, -1)
            x_vec_scaled = self.sgd_scalers_[target].transform(x_vec)

            model = self.sgd_models_[target]
            if target in CAT_COLS:
                model.partial_fit(x_vec_scaled, y_val, classes=model.classes_)
            else:
                model.partial_fit(x_vec_scaled, y_val)

            self.sgd_samples_seen_[target] += 1
            updated.append(target)

        if verbose:
            for t in updated:
                seen = self.sgd_samples_seen_[t]
                active = "✓ ACTIF" if seen >= self.sgd_trust_threshold else f"en cours ({seen}/{self.sgd_trust_threshold})"
                print(f"  {t:<32} {active}")


    # ───────────────────────────────────────────────────────────────────────────
    # Métriques et état
    # ───────────────────────────────────────────────────────────────────────────

    def scores(self) -> pd.DataFrame:
        """Scores RF par colonne cible."""
        self._check_fitted()
        rows = [{"colonne": col, **info} for col, info in self.rf_metrics_.items()]
        return pd.DataFrame(rows).set_index("colonne").sort_values("score", ascending=False)

    def sgd_status(self) -> pd.DataFrame:
        """État de la couche SGD : exemples vus et source active par colonne."""
        if not self._sgd_ready:
            print("SGD non initialisé.")
            return pd.DataFrame()
        rows = []
        for target in self.target_cols_:
            seen   = self.sgd_samples_seen_.get(target, 0)
            active = seen >= self.sgd_trust_threshold
            rows.append({
                "colonne":      target,
                "exemples_vus": seen,
                "seuil":        self.sgd_trust_threshold,
                "source_active": "SGD ✓" if active else f"RF  ({seen}/{self.sgd_trust_threshold})",
            })
        return pd.DataFrame(rows).set_index("colonne")


    # ───────────────────────────────────────────────────────────────────────────
    # Sauvegarde / Chargement
    # ───────────────────────────────────────────────────────────────────────────

    def save(self, path: str = "student_model.pkl") -> None:
        self._check_fitted()
        with open(path, "wb") as f:
            pickle.dump({
                "rf_models":        self.rf_models_,
                "rf_metrics":       self.rf_metrics_,
                "sgd_models":       self.sgd_models_,
                "sgd_scalers":      self.sgd_scalers_,
                "sgd_samples_seen": self.sgd_samples_seen_,
                "sgd_features":     self.sgd_features_,
                "all_cols":         self.all_cols_,
                "target_cols":      self.target_cols_,
                "params": {
                    "n_estimators":        self.n_estimators,
                    "max_depth":           self.max_depth,
                    "sgd_trust_threshold": self.sgd_trust_threshold,
                },
            }, f)
        print(f"Modèle sauvegardé → {path}")

    def load(self, path: str = "student_model.pkl") -> "StudentPredictor":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.rf_models_        = data["rf_models"]
        self.rf_metrics_       = data["rf_metrics"]
        self.sgd_models_       = data.get("sgd_models", {})
        self.sgd_scalers_      = data.get("sgd_scalers", {})
        self.sgd_samples_seen_ = data.get("sgd_samples_seen", {})
        self.sgd_features_     = data.get("sgd_features", {})
        self.all_cols_         = data["all_cols"]
        self.target_cols_      = data["target_cols"]
        p = data["params"]
        self.n_estimators        = p["n_estimators"]
        self.max_depth           = p["max_depth"]
        self.sgd_trust_threshold = p.get("sgd_trust_threshold", SGD_TRUST_THRESHOLD)
        self._fitted    = True
        self._sgd_ready = bool(self.sgd_models_)
        print(f"Modèle chargé depuis {path}")
        return self


    # ───────────────────────────────────────────────────────────────────────────
    # Méthodes privées
    # ───────────────────────────────────────────────────────────────────────────

    def _rf_predict(self, target: str, known: dict):
        model    = self.rf_models_[target]
        features = [c for c in self.all_cols_ if c != target]
        x = np.array([
            known.get(f, 0.0) if known.get(f) is not None else 0.0
            for f in features
        ]).reshape(1, -1)
        pred_val = model.predict(x)[0]
        conf = None
        if target in CAT_COLS:
            conf = max(model.predict_proba(x)[0])
        return pred_val, conf, "RF"

    def _sgd_predict(self, target: str, known: dict):
        """
        Vecteur d'entrée SGD :
          - Ancres → vraies valeurs fournies (jamais estimées)
          - Autres cibles → valeur fournie si disponible, sinon prédiction RF
        """
        model        = self.sgd_models_[target]
        sgd_features = self.sgd_features_[target]

        x_vals = []
        for f in sgd_features:
            if f in ANCHOR_COLS:
                x_vals.append(known.get(f, 0.0))
            elif f in known and known[f] is not None:
                x_vals.append(known[f])
            else:
                rf_pred, _, _ = self._rf_predict(f, known)
                x_vals.append(rf_pred)

        x = np.array(x_vals).reshape(1, -1)
        x_scaled = self.sgd_scalers_[target].transform(x)
        pred_val = model.predict(x_scaled)[0]
        conf = None
        if target in CAT_COLS and hasattr(model, 'predict_proba'):
            try:
                conf = max(model.predict_proba(x_scaled)[0])
            except Exception:
                pass
        return pred_val, conf, "SGD"

    def _build_sgd_X(self, df: pd.DataFrame,
                     target: str, sgd_features: list) -> np.ndarray:
        """
        Construit la matrice X pour l'init SGD — version vectorisée.
        Toutes les prédictions RF sont faites en batch (une seule fois par cible).
        """
        rf_batch = self._batch_rf_predict(df)
        cols = []
        for f in sgd_features:
            if f in ANCHOR_COLS:
                cols.append(df[f].values)
            else:
                cols.append(rf_batch[f])
        return np.column_stack(cols)

    def _batch_rf_predict(self, df: pd.DataFrame) -> dict:
        """Prédit toutes les cibles RF en batch. Retourne {col: array}."""
        result = {}
        for t in self.target_cols_:
            feats     = [c for c in self.all_cols_ if c != t]
            result[t] = self.rf_models_[t].predict(df[feats].values)
        return result

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Modèle non entraîné. Appelle .fit() d'abord.")