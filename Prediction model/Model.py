import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import xgboost

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QScrollArea, QGroupBox,
    QDoubleSpinBox, QComboBox, QMessageBox,
    QProgressBar, QFrame
)

# ====== 路径：改成你自己的 ======
def resource_path(*relative_parts):
    """
    开发时：基于脚本所在目录
    打包后：基于 exe 所在目录（--onedir）或 PyInstaller 解包目录（--onefile）
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后的临时/运行目录
        base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, *relative_parts)

MODEL_BUNDLE_PATH = resource_path("model_files", "xgb_best_bundle.joblib")
FEATURE_NAMES_PATH = resource_path("repro_data", "final_15_feature_names.json")

# ====== 定义字段类型 ======
continuous_fields = {
    "EORTC QLQ-C30_Physical functioning": (0, 100),
    "EORTC QLQ-C30_Cognitive functioning": (0, 100),
    "EORTC QLQ-C30_Social functioning": (0, 100),
    "EORTC QLQ-C30_Quality of life": (0, 100),
    "PSSS_Significant other support": (4, 28),
    "PAIS-SR_Health care orientation": (0, 21),
    "PAIS-SR_Vocational environment": (0, 18),
    "PAIS-SR_Domestic environment": (0, 21),
    "PAIS-SR_Extended family relationships": (0, 15),
    "PAIS-SR_Psychological distress": (0, 21),
}

categorical_fields = {
    "HADS_Anxiety": [0, 1, 2, 3],
    "Self-care ability": [1, 2, 3, 4],
    "Nutritional risk": [0, 1],
    "Religious belief": [0, 1],
    "Treatment modality_Chemotherapy": [0, 1],
}

field_help = {
    "HADS_Anxiety": "0 None  1 Mild  2 Moderate  3 Severe",
    "Self-care ability": "1 No dependence  2 Mild  3 Moderate  4 Severe",
    "Nutritional risk": "0 Risk-free  1 There is a risk",
    "Religious belief": "0 No  1 Yes",
    "Treatment modality_Chemotherapy": "0 No  1 Yes",
}


def load_assets():
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    model = bundle["model"]
    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    return model, feature_names


def set_app_style(app: QApplication):
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet("""
        QMainWindow { background: #F3F4F6; }
        QLabel#Title { font-size: 20px; font-weight: 700; color: #111827; }
        QLabel#Subtitle { color: #6B7280; }

        QFrame#Card {
            background: white;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
        }

        QGroupBox {
            font-weight: 700;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            margin-top: 10px;
            padding: 10px;
            background: #FFFFFF;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            top: 18px;              /* ⭐ 让标题往下移动 */
            padding: 0 6px;
            color: #111827;
        }

        QPushButton {
            border-radius: 10px;
            padding: 10px 14px;
            font-weight: 600;
        }
        QPushButton#Primary { background: #2563EB; color: white; }
        QPushButton#Primary:hover { background: #1D4ED8; }
        QPushButton#Secondary {
            background: #F9FAFB;
            color: #2563EB;
            border: 1px solid #E5E7EB;
        }
        QPushButton#Secondary:hover { background: #F3F4F6; }

        QDoubleSpinBox, QComboBox {
            padding: 6px 10px;
            border-radius: 10px;
            border: 1px solid #E5E7EB;
            background: #FFFFFF;
            min-height: 30px;
        }

        QProgressBar {
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            text-align: center;
            background: #F9FAFB;
            height: 18px;
        }
        QProgressBar::chunk {
            background: #2563EB;
            border-radius: 10px;
        }
    """)


class PredictorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lung cancer core symptom cluster prediction model")
        self.resize(1000, 720)

        try:
            self.model, self.feature_names = load_assets()
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to load model/resources:\n{e}")
            raise

        # widgets: fname -> ("continuous"/"categorical", widget)
        self.widgets = {}

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(12)

        title = QLabel("Lung cancer core symptom cluster prediction model")
        title.setObjectName("Title")
        subtitle = QLabel("Desktop version (PySide6). Input features and click Predict.")
        subtitle.setObjectName("Subtitle")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        # Card container
        card = QFrame()
        card.setObjectName("Card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(12)
        outer.addWidget(card, 1)

        # Scroll area for inputs
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        card_layout.addWidget(scroll, 1)

        scroll_body = QWidget()
        scroll.setWidget(scroll_body)
        body_layout = QVBoxLayout(scroll_body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(12)

        # Only two groups now
        body_layout.addWidget(self._build_continuous_group())
        body_layout.addWidget(self._build_categorical_group())

        # Result card
        self.result_card = QFrame()
        self.result_card.setObjectName("Card")
        res_layout = QVBoxLayout(self.result_card)
        res_layout.setContentsMargins(16, 16, 16, 16)
        res_layout.setSpacing(10)
        outer.addWidget(self.result_card)

        self.result_label = QLabel("The prediction results will be displayed here")
        self.result_label.setStyleSheet("font-size: 14px; font-weight: 700; color: #111827;")
        res_layout.addWidget(self.result_label)

        self.pbars = {}
        self.proba_text = {}
        for name in ["Low", "Medium", "High"]:
            row = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setMinimumWidth(70)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)

            txt = QLabel("0.000")
            txt.setMinimumWidth(60)
            txt.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            self.pbars[name] = bar
            self.proba_text[name] = txt

            row.addWidget(lbl)
            row.addWidget(bar, 1)
            row.addWidget(txt)
            res_layout.addLayout(row)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self.btn_predict = QPushButton("Predict")
        self.btn_predict.setObjectName("Primary")
        self.btn_predict.clicked.connect(self.do_predict)

        self.btn_clear = QPushButton("Clear out")
        self.btn_clear.setObjectName("Secondary")
        self.btn_clear.clicked.connect(self.clear_all)

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.setObjectName("Secondary")
        self.btn_exit.clicked.connect(self.close)

        btn_row.addWidget(self.btn_predict)
        btn_row.addWidget(self.btn_clear)
        btn_row.addWidget(self.btn_exit)
        outer.addLayout(btn_row)

        # Safety: ensure all feature_names are covered by your dictionaries
        self._assert_all_fields_covered()

    def _assert_all_fields_covered(self):
        defined = set(continuous_fields.keys()) | set(categorical_fields.keys())
        missing = [f for f in self.feature_names if f not in defined]
        if missing:
            QMessageBox.critical(
                self,
                "Field definition error",
                "Some fields in feature_names are not defined in continuous_fields/categorical_fields:\n"
                + "\n".join(missing)
            )
            raise ValueError("Undefined fields exist in feature_names")

    def _build_continuous_group(self) -> QGroupBox:
        box = QGroupBox("Continuous variables")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        r = 0
        for fname in self.feature_names:
            if fname in continuous_fields:
                mn, mx = continuous_fields[fname]

                name_lbl = QLabel(fname)
                spin = QDoubleSpinBox()
                spin.setDecimals(3)
                spin.setRange(float(mn), float(mx))
                spin.setSingleStep(0.1)
                spin.setValue(float(mn))  # 默认最小值（你想改默认值我也可以帮你改）

                help_lbl = QLabel(f"Range: {mn} - {mx}")
                help_lbl.setStyleSheet("color:#6B7280; font-size:11px;")

                self.widgets[fname] = ("continuous", spin)

                grid.addWidget(name_lbl, r, 0, Qt.AlignLeft)
                grid.addWidget(spin, r, 1)
                grid.addWidget(help_lbl, r, 2, Qt.AlignLeft)
                r += 1

        grid.setColumnStretch(0, 4)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 3)
        return box

    def _build_categorical_group(self) -> QGroupBox:
        box = QGroupBox("Categorical variables")
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        r = 0
        for fname in self.feature_names:
            if fname in categorical_fields:
                name_lbl = QLabel(fname)

                cb = QComboBox()
                for v in categorical_fields[fname]:
                    cb.addItem(str(v), int(v))
                cb.setCurrentIndex(0)

                help_lbl = QLabel(field_help.get(fname, ""))
                help_lbl.setStyleSheet("color:#6B7280; font-size:11px;")

                self.widgets[fname] = ("categorical", cb)

                grid.addWidget(name_lbl, r, 0, Qt.AlignLeft)
                grid.addWidget(cb, r, 1)
                grid.addWidget(help_lbl, r, 2, Qt.AlignLeft)
                r += 1

        grid.setColumnStretch(0, 4)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 3)
        return box

    def parse_inputs(self) -> np.ndarray:
        vals = []

        # 需要做 100-原值 的字段（按你的 feature_names 命名保持一致）
        reverse_100_fields = {
            "EORTC QLQ-C30_Physical functioning",
            "EORTC QLQ-C30_Cognitive functioning",
            "EORTC QLQ-C30_Social functioning",
        }

        for fname in self.feature_names:
            ftype, w = self.widgets[fname]

            if ftype == "categorical":
                v = float(int(w.currentData()))
            else:
                v = float(w.value())
                if fname in reverse_100_fields:
                    v = 100.0 - v  # ✅ 关键：转换后再进模型

            vals.append(v)

        return np.array(vals, dtype=float)

    def do_predict(self):
        try:
            x = self.parse_inputs()
            X = pd.DataFrame([x], columns=self.feature_names)

            pred = int(self.model.predict(X)[0])
            proba = self.model.predict_proba(X)[0]

            categories = ["Low", "Medium", "High"]
            self.result_label.setText(f"Prediction category: {categories[pred]}")

            self._set_bar("Low", float(proba[0]))
            self._set_bar("Medium", float(proba[1]))
            self._set_bar("High", float(proba[2]))

        except Exception as e:
            QMessageBox.critical(self, "Predict error", str(e))

    def _set_bar(self, name: str, p: float):
        p = max(0.0, min(1.0, p))
        self.pbars[name].setValue(int(round(p * 100)))
        self.proba_text[name].setText(f"{p:.3f}")

    def clear_all(self):
        for fname in self.feature_names:
            ftype, w = self.widgets[fname]
            if ftype == "categorical":
                w.setCurrentIndex(0)
            else:
                mn, _ = continuous_fields[fname]
                w.setValue(float(mn))

        self.result_label.setText("The prediction results will be displayed here")
        for k in ["Low", "Medium", "High"]:
            self._set_bar(k, 0.0)


def main():
    app = QApplication(sys.argv)
    set_app_style(app)

    win = PredictorWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
