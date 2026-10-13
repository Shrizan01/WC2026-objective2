"""Checks that directly protect against temporal and tuning leakage."""
import sys
from pathlib import Path
import unittest
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from run_analysis import date_splits, estimator

class ValidationTests(unittest.TestCase):
    def test_dates_do_not_cross_folds(self):
        dates = pd.Series(pd.date_range('2026-06-01', periods=20).repeat(3))
        for tr, va in date_splits(dates):
            self.assertLess(dates.iloc[tr].max(), dates.iloc[va].min())
            self.assertFalse(set(dates.iloc[tr]) & set(dates.iloc[va]))
            self.assertEqual(len(va) % 3, 0)

    def test_scaler_is_inside_tuned_estimator(self):
        dates = pd.Series(pd.date_range('2026-06-01', periods=20))
        for name in ['Ridge','Lasso']:
            model = estimator(name, dates)
            self.assertIsInstance(model, GridSearchCV)
            self.assertIsInstance(model.estimator, Pipeline)
            self.assertEqual(model.estimator.steps[0][0], 'standardscaler')
            for tr, va in model.cv:
                self.assertLess(dates.iloc[tr].max(), dates.iloc[va].min())

if __name__ == '__main__':
    unittest.main()
