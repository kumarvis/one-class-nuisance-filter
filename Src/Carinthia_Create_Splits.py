"""Create leakage-safe train/validation/test folders for the Carinthia dataset.

Expected input layout (as in the supplied folder screenshot):

    project_root/
    ├── Carinthia_Create_Splits.py
    └── Data/
        ├── carinthia.csv
        └── images/
            └── <image files>

The input CSV is expected to use semicolon separators and contain
image_path, file_name, and label columns.
"""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
CSV_PATH = DATA_DIR / "carinthia.csv"

CSV_SEPARATOR = ";"
RANDOM_SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

SPLIT_NAMES = ("train", "validation", "test")
EXPECTED_COLUMNS = {"image_path", "file_name", "label"}


# -----------------------------------------------------------------------------
# Input validation and image path resolution
# -----------------------------------------------------------------------------
def load_metadata(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Set CSV_PATH near the top of this script to the correct location."
        )

    frame = pd.read_csv(csv_path, sep=CSV_SEPARATOR, dtype={"label": "string"})
    frame.columns = frame.columns.str.strip()

    for column in frame.select_dtypes(include=["object", "string"]).columns:
        frame[column] = frame[column].str.strip()

    missing_columns = EXPECTED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing_columns)}. "
            f"Columns found: {frame.columns.tolist()}"
        )

    if frame[list(EXPECTED_COLUMNS)].isna().any().any():
        bad_rows = frame.loc[
            frame[list(EXPECTED_COLUMNS)].isna().any(axis=1),
            list(EXPECTED_COLUMNS),
        ]
        raise ValueError(
            "Required CSV fields contain missing values. Fix these rows first:\n"
            f"{bad_rows.head(10).to_string(index=False)}"
        )

    return frame


def resolve_source_image(row: pd.Series) -> Path:
    """Resolve CSV paths such as data/images/name.jpg under Data/images/name.jpg."""
    raw_path = str(row["image_path"]).replace("\\", "/")
    relative_path = Path(raw_path)
    file_name = Path(str(row["file_name"]).replace("\\", "/")).name

    candidates = [
        # Handles CSV paths already relative to Data/.
        DATA_DIR / relative_path,
    ]

    # In the supplied folder layout, the CSV says data/images/name.jpg,
    # while the extracted files are Data/images/name.jpg. Drop the first
    # "data" component when resolving against DATA_DIR.
    if relative_path.parts and relative_path.parts[0].lower() == "data":
        candidates.append(DATA_DIR.joinpath(*relative_path.parts[1:]))

    candidates.extend([
        DATA_DIR / "images" / file_name,
        CSV_PATH.parent / relative_path,
        CSV_PATH.parent / "images" / file_name,
    ])

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find image for CSV row:\n"
        f"  image_path={row['image_path']}\n"
        f"  file_name={row['file_name']}\n"
        f"Paths tried:\n{tried}"
    )


# -----------------------------------------------------------------------------
# Deterministic, approximately stratified split
# -----------------------------------------------------------------------------
def assign_splits(frame: pd.DataFrame) -> pd.Series:
    fractions = np.array(
        [TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION], dtype=float
    )
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Train, validation, and test fractions must sum to 1.0.")

    rng = np.random.default_rng(RANDOM_SEED)
    split_by_index = {}

    # Labels are used only to distribute examples across partitions.
    # BYOL will receive images without labels, and Deep SVDD will later
    # receive only label 3 from the train partition.
    for _, class_rows in frame.groupby("label", sort=True, dropna=False):
        indices = class_rows.index.to_numpy().copy()
        rng.shuffle(indices)
        count = len(indices)

        if count == 1:
            train_count, validation_count, test_count = 0, 0, 1
        elif count == 2:
            train_count, validation_count, test_count = 0, 1, 1
        else:
            validation_count = max(1, int(round(count * VALIDATION_FRACTION)))
            test_count = max(1, int(round(count * TEST_FRACTION)))

            # Keep at least one training example for every class with 3+ rows.
            if validation_count + test_count >= count:
                validation_count = 1
                test_count = 1

            train_count = count - validation_count - test_count

        train_end = train_count
        validation_end = train_count + validation_count

        for index in indices[:train_end]:
            split_by_index[index] = "train"
        for index in indices[train_end:validation_end]:
            split_by_index[index] = "validation"
        for index in indices[validation_end:]:
            split_by_index[index] = "test"

    return pd.Series(split_by_index, name="split").reindex(frame.index)


# -----------------------------------------------------------------------------
# Output generation
# -----------------------------------------------------------------------------
def ensure_output_folders_are_empty(data_dir: Path) -> None:
    existing_nonempty = []
    for split_name in SPLIT_NAMES:
        split_dir = data_dir / split_name
        if split_dir.exists() and any(split_dir.iterdir()):
            existing_nonempty.append(split_dir)

    if existing_nonempty:
        folders = "\n".join(f"  - {folder}" for folder in existing_nonempty)
        raise FileExistsError(
            "Output split folder(s) already contain files. Move or rename them "
            "before running this script to avoid mixing old and new splits:\n"
            f"{folders}"
        )


def write_split_folders(frame: pd.DataFrame, data_dir: Path) -> None:
    for split_name in SPLIT_NAMES:
        split_dir = data_dir / split_name
        image_dir = split_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        split_rows = frame.loc[frame["split"] == split_name].copy()
        ground_truth_rows = []
        metadata_rows = []

        for original_index, row in split_rows.iterrows():
            source_path = Path(row["resolved_image_path"])
            destination_path = image_dir / Path(str(row["file_name"])).name
            shutil.copy2(source_path, destination_path)

            split_image_path = Path("images") / destination_path.name
            ground_truth_rows.append({
                "image_path": split_image_path.as_posix(),
                "file_name": destination_path.name,
                "label": str(row["label"]),
            })
            metadata_rows.append({
                "original_row_index": original_index,
                "source_image_path": str(row["image_path"]),
                "split_image_path": split_image_path.as_posix(),
                "file_name": destination_path.name,
                "label": str(row["label"]),
            })

        pd.DataFrame(ground_truth_rows).to_csv(
            split_dir / "ground_truth.csv",
            sep=CSV_SEPARATOR,
            index=False,
        )
        pd.DataFrame(metadata_rows).to_csv(
            split_dir / "metadata.csv",
            sep=CSV_SEPARATOR,
            index=False,
        )

        print(f"{split_name:10s}: {len(split_rows):,} rows -> {split_dir}")


def main() -> None:
    metadata = load_metadata(CSV_PATH)

    # Avoid duplicate filenames causing files to be overwritten or the same
    # image to appear in more than one partition.
    duplicate_names = metadata["file_name"].duplicated(keep=False)
    if duplicate_names.any():
        examples = metadata.loc[duplicate_names, ["image_path", "file_name", "label"]]
        raise ValueError(
            "Duplicate file_name values were found. Resolve them before splitting "
            "to prevent overwriting or cross-split leakage. Examples:\n"
            f"{examples.head(10).to_string(index=False)}"
        )

    metadata["resolved_image_path"] = metadata.apply(resolve_source_image, axis=1)
    metadata["split"] = assign_splits(metadata)

    if metadata["split"].isna().any():
        raise RuntimeError("Some rows did not receive a split assignment.")

    ensure_output_folders_are_empty(DATA_DIR)
    write_split_folders(metadata, DATA_DIR)

    summary = (
        metadata.groupby(["split", "label"], observed=True)
        .size()
        .rename("image_count")
        .reset_index()
    )
    summary.to_csv(DATA_DIR / "split_summary.csv", sep=CSV_SEPARATOR, index=False)

    print("\nSplit summary:")
    print(summary.pivot(index="label", columns="split", values="image_count").fillna(0).astype(int))
    print(f"\nSaved split summary to: {DATA_DIR / 'split_summary.csv'}")
    print("\nFor the agreed pipeline:")
    print("- BYOL: use images from Data/train/images, without reading ground_truth.csv.")
    print("- Deep SVDD: use only label 3 rows from Data/train/ground_truth.csv.")
    print("- Validation/test: keep out of both training stages.")


if __name__ == "__main__":
    main()
