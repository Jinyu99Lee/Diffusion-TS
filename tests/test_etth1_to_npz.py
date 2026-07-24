import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Data import etth1_to_npz


class ETTh1ToNPZTest(unittest.TestCase):
    def _frame(self, rows=etth1_to_npz.TRAIN_POINTS):
        row_ids = np.arange(rows, dtype=np.float64)
        payload = {
            column: row_ids + (feature_id + 1) * 10_000
            for feature_id, column in enumerate(etth1_to_npz.FEATURE_COLS)
        }
        # Deliberately reverse CSV feature order; output must use FEATURE_COLS.
        payload = {
            "date": pd.date_range("2016-07-01", periods=rows, freq="h"),
            **{column: payload[column] for column in reversed(etth1_to_npz.FEATURE_COLS)},
        }
        return pd.DataFrame(payload)

    def _write(self, directory, frame, name="ETTh1.csv"):
        path = Path(directory) / name
        frame.to_csv(path, index=False)
        return path

    def test_default_horizon_counts_and_disjoint_exhaustive_indices(self):
        expected = {
            96: (8209, 6978, 1231),
            192: (8113, 6896, 1217),
            336: (7969, 6774, 1195),
            720: (7585, 6447, 1138),
        }
        for horizon, (expected_n, expected_train, expected_val) in expected.items():
            with self.subTest(horizon=horizon):
                seq_len = 336 + horizon
                n = etth1_to_npz._num_windows(8640, seq_len, stride=1)
                train, val, start, end = etth1_to_npz._split_sample_indices(
                    n, 0.70, 0.85
                )
                self.assertEqual(n, expected_n)
                self.assertEqual(len(train), expected_train)
                self.assertEqual(len(val), expected_val)
                self.assertEqual(start, int(np.floor(0.70 * n)))
                self.assertEqual(end, int(np.floor(0.85 * n)))
                self.assertEqual(len(np.intersect1d(train, val)), 0)
                np.testing.assert_array_equal(
                    np.sort(np.concatenate((train, val))), np.arange(n)
                )

    def test_conversion_preserves_feature_order_values_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = self._frame(etth1_to_npz.TRAIN_POINTS + 1)
            # Invalid data outside the first 8,640 rows must never be consumed.
            frame.loc[etth1_to_npz.TRAIN_POINTS, "date"] = "not-a-date"
            frame.loc[etth1_to_npz.TRAIN_POINTS, "OT"] = np.inf
            csv_path = self._write(temporary, frame)
            output_dir = Path(temporary) / "output"

            [(train_path, val_path, meta_path)] = etth1_to_npz.convert(
                csv_path=csv_path,
                output_dir=output_dir,
                lookback=4,
                horizons=[2],
                val_start_ratio=0.70,
                val_end_ratio=0.85,
                stride=2,
            )

            self.assertEqual(
                train_path,
                output_dir / "T6" / "etth1_T6_p2_train.npz",
            )
            self.assertEqual(
                val_path,
                output_dir / "T6" / "etth1_T6_p2_val.npz",
            )
            self.assertEqual(
                meta_path,
                output_dir / "T6" / "etth1_T6_p2_meta.json",
            )

            n = (8640 - 6) // 2 + 1
            val_start = int(np.floor(0.70 * n))
            val_end = int(np.floor(0.85 * n))
            with np.load(train_path, allow_pickle=False) as train_npz, np.load(
                val_path, allow_pickle=False
            ) as val_npz:
                train = train_npz["data"]
                val = val_npz["data"]
                train_indices = train_npz["sample_indices"]
                val_indices = val_npz["sample_indices"]

                self.assertEqual(train.dtype, np.float32)
                self.assertEqual(val.dtype, np.float32)
                self.assertEqual(train.shape, (n - (val_end - val_start), 6, 7))
                self.assertEqual(val.shape, (val_end - val_start, 6, 7))
                np.testing.assert_array_equal(
                    train_npz["feature_cols"],
                    np.asarray(etth1_to_npz.FEATURE_COLS),
                )
                np.testing.assert_array_equal(
                    np.sort(np.concatenate((train_indices, val_indices))),
                    np.arange(n),
                )
                self.assertEqual(
                    len(np.intersect1d(train_indices, val_indices)), 0
                )
                np.testing.assert_array_equal(
                    train_npz["window_start_indices"], train_indices * 2
                )
                np.testing.assert_array_equal(
                    val_npz["window_start_indices"], val_indices * 2
                )

                expected_first_row = np.arange(1, 8) * 10_000
                np.testing.assert_array_equal(train[0, 0], expected_first_row)
                np.testing.assert_array_equal(
                    val[0, 0], expected_first_row + val_start * 2
                )
                embedded = json.loads(str(val_npz["meta"]))
                self.assertEqual(embedded["split"], "val")
                self.assertEqual(embedded["split_samples"], len(val_indices))

            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["N"], n)
            self.assertEqual(metadata["source_row_range"], [0, 8640])
            self.assertEqual(metadata["feature_cols"], list(etth1_to_npz.FEATURE_COLS))
            self.assertEqual(metadata["lookback"], 4)
            self.assertEqual(metadata["pred_len"], 2)
            self.assertEqual(metadata["target_delay"], 0)
            self.assertEqual(metadata["layout"], "aligned")
            self.assertEqual(metadata["val_sample_start"], val_start)
            self.assertEqual(metadata["val_sample_end_exclusive"], val_end)
            self.assertEqual(
                metadata["source_last_timestamp"], "2017-06-25T23:00:00"
            )

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                etth1_to_npz.convert(
                    csv_path=csv_path,
                    output_dir=output_dir,
                    lookback=4,
                    horizons=[2],
                    stride=2,
                )

    def test_rejects_invalid_training_prefixes(self):
        cases = {}

        duplicate = self._frame()
        duplicate.loc[100, "date"] = duplicate.loc[99, "date"]
        cases["duplicate timestamps"] = duplicate

        gap = self._frame()
        gap.loc[100:, "date"] = gap.loc[100:, "date"] + pd.Timedelta(hours=1)
        cases["hourly-contiguous"] = gap

        missing_value = self._frame()
        missing_value.loc[100, "HUFL"] = np.nan
        cases["missing or non-numeric"] = missing_value

        infinite_value = self._frame()
        infinite_value.loc[100, "OT"] = np.inf
        cases["non-finite"] = infinite_value

        invalid_date = self._frame()
        invalid_date.loc[100, "date"] = "bad-date"
        cases["missing or invalid timestamps"] = invalid_date

        missing_column = self._frame().drop(columns=["LUFL"])
        cases["missing required columns"] = missing_column

        too_short = self._frame(etth1_to_npz.TRAIN_POINTS - 1)
        cases["at least 8640"] = too_short

        with tempfile.TemporaryDirectory() as temporary:
            for case_id, (message, frame) in enumerate(cases.items()):
                with self.subTest(message=message):
                    path = self._write(
                        temporary, frame, name=f"invalid_{case_id}.csv"
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        etth1_to_npz._load_training_prefix(path)

    def test_window_view_uses_stride_without_copying_wrong_axis(self):
        values = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)
        windows = etth1_to_npz._window_view(values, seq_len=3, stride=2)
        self.assertEqual(windows.shape, (3, 3, 2))
        np.testing.assert_array_equal(windows[0], values[0:3])
        np.testing.assert_array_equal(windows[1], values[2:5])
        np.testing.assert_array_equal(windows[2], values[4:7])


if __name__ == "__main__":
    unittest.main()
