from pathlib import Path
import argparse
import sys
import json
import hashlib
import re

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    # ============================================================
    # 0. 基本配置
    # ============================================================

    ROOT = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Prepare or reproduce the fixed 5000-patient MIMIC subset.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "subset",
                        help="Destination inside this project; use a separate directory to verify reproducibility")
    args = parser.parse_args()
    RAW_DIR = ROOT / "raw"
    OUT_DIR = args.output_dir.resolve()
    OUT_DIR.relative_to(ROOT)

    META_PATH = RAW_DIR / "mimic-cxr-2.0.0-metadata.csv.gz"
    CHEX_PATH = RAW_DIR / "mimic-cxr-2.0.0-chexpert.csv.gz"

    # 解压 mimic-cxr-reports.zip 后，应指向真正包含 p10、p11... 的 files 目录
    REPORT_ROOT = RAW_DIR / "reports" / "files"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    N_PATIENTS = 5000
    SEED = 42

    DISEASES = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
    ]

    BASE_URL = (
        "https://physionet.org/files/"
        "mimic-cxr-jpg/2.1.0/files"
    )


    # ============================================================
    # 1. 基础文件检查
    # ============================================================

    print("=" * 70)
    print("Step 1: Checking input files")
    print("=" * 70)

    required_paths = [
        META_PATH,
        CHEX_PATH,
        REPORT_ROOT,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path does not exist:\n{path}"
            )

    print("Metadata:", META_PATH)
    print("CheXpert:", CHEX_PATH)
    print("Reports:", REPORT_ROOT)

    print("\nInput file check passed.")


    # ============================================================
    # 2. 读取 metadata
    # ============================================================

    print("\n" + "=" * 70)
    print("Step 2: Loading metadata")
    print("=" * 70)

    meta = pd.read_csv(META_PATH)

    required_meta_columns = [
        "subject_id",
        "study_id",
        "dicom_id",
        "ViewPosition",
    ]

    missing_columns = [
        col
        for col in required_meta_columns
        if col not in meta.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {missing_columns}"
        )

    meta["subject_id"] = meta["subject_id"].astype("int64")
    meta["study_id"] = meta["study_id"].astype("int64")
    meta["dicom_id"] = meta["dicom_id"].astype(str)

    print(f"Total metadata rows: {len(meta):,}")
    print(
        f"Total patients: "
        f"{meta['subject_id'].nunique():,}"
    )


    # ============================================================
    # 3. 只保留 AP / PA 正位胸片
    # ============================================================

    print("\n" + "=" * 70)
    print("Step 3: Filtering AP / PA frontal images")
    print("=" * 70)

    # 严格按指导书先固定每位患者最小 study_id 的检查，再筛 AP/PA。
    # 最小 study_id 检查没有正位图像时剔除患者，不回退到其他检查。
    minimum_study = meta.groupby("subject_id")["study_id"].transform("min")
    meta = meta[meta["study_id"] == minimum_study].copy()

    meta = meta[
        meta["ViewPosition"].isin(["AP", "PA"])
    ].copy()

    print(
        f"AP / PA images: {len(meta):,}"
    )

    print(
        f"Patients with AP / PA images: "
        f"{meta['subject_id'].nunique():,}"
    )

    if len(meta) == 0:
        raise RuntimeError(
            "No AP / PA images found."
        )


    # ============================================================
    # 4. 读取并合并 CheXpert 标签
    # ============================================================

    print("\n" + "=" * 70)
    print("Step 4: Loading CheXpert labels")
    print("=" * 70)

    chex = pd.read_csv(CHEX_PATH)

    required_chex_columns = [
        "subject_id",
        "study_id",
    ] + DISEASES

    missing_columns = [
        col
        for col in required_chex_columns
        if col not in chex.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing CheXpert columns: {missing_columns}"
        )

    chex["subject_id"] = chex["subject_id"].astype("int64")
    chex["study_id"] = chex["study_id"].astype("int64")

    chex = chex[
        ["subject_id", "study_id"] + DISEASES
    ].copy()

    meta = meta.merge(
        chex,
        on=["subject_id", "study_id"],
        how="left",
        validate="many_to_one",
    )

    print(
        f"Merged rows: {len(meta):,}"
    )


    # ============================================================
    # 5. 报告解析函数
    # ============================================================

    print("\n" + "=" * 70)
    print("Step 5: Preparing report parser")
    print("=" * 70)

    REPORT_HEADINGS = (
        r"FINDINGS|"
        r"IMPRESSION|"
        r"HISTORY|"
        r"INDICATION|"
        r"COMPARISON|"
        r"TECHNIQUE|"
        r"EXAMINATION|"
        r"REASON FOR EXAMINATION|"
        r"CLINICAL HISTORY"
    )

    FINDINGS_PATTERN = re.compile(
        rf"(?ims)"
        rf"^\s*FINDINGS\s*:\s*"
        rf"(.*?)"
        rf"(?=^\s*(?:{REPORT_HEADINGS})\s*:|\Z)"
    )

    IMPRESSION_PATTERN = re.compile(
        rf"(?ims)"
        rf"^\s*IMPRESSION\s*:\s*"
        rf"(.*?)"
        rf"(?=^\s*(?:{REPORT_HEADINGS})\s*:|\Z)"
    )


    def get_report_path(subject_id, study_id):
        """
        Example:
        subject_id = 10000032
        study_id   = 50414267

        path:
        files/p10/p10000032/s50414267.txt
        """

        subject_id = str(int(subject_id))
        study_id = str(int(study_id))

        prefix = "p" + subject_id[:2]

        return (
            REPORT_ROOT
            / prefix
            / f"p{subject_id}"
            / f"s{study_id}.txt"
        )


    def load_report(subject_id, study_id):
        """
        Project rule:

        1. Prefer Findings.
        2. If Findings is unavailable, use Impression.
        3. If both are empty, return None.
        """

        report_path = get_report_path(
            subject_id,
            study_id,
        )

        try:
            text = report_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        except FileNotFoundError:
            return None

        except Exception:
            return None

        findings_match = FINDINGS_PATTERN.search(text)

        if findings_match:
            findings = findings_match.group(1).strip()

            if findings:
                return findings

        impression_match = IMPRESSION_PATTERN.search(text)

        if impression_match:
            impression = impression_match.group(1).strip()

            if impression:
                return impression

        return None


    print("Report parser ready.")


    # ============================================================
    # 6. 每位患者按 study_id 排序，只选择一次检查
    # ============================================================

    print("\n" + "=" * 70)
    print("Step 6: Selecting one study per patient")
    print("=" * 70)

    # 首先排序
    meta = meta.sort_values(
        [
            "subject_id",
            "study_id",
            "dicom_id",
        ]
    ).reset_index(drop=True)

    # 每位患者已在视角筛选前固定其全部检查中最小的 study_id
    first_study_id = (
        meta
        .groupby(
            "subject_id",
            sort=False,
        )["study_id"]
        .transform("min")
    )

    first_study = meta[
        meta["study_id"] == first_study_id
    ].copy()

    # 一个 study 可能存在多张 AP / PA
    # 按 dicom_id 排序后只保留第一张
    candidates = (
        first_study
        .sort_values(
            [
                "subject_id",
                "study_id",
                "dicom_id",
            ]
        )
        .drop_duplicates(
            subset=["subject_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        "Patients with one candidate AP / PA study: "
        f"{len(candidates):,}"
    )

    # ============================================================
    # 7. 读取报告并筛选有效患者
    # ============================================================

    print("\n" + "=" * 70)
    print("Step 7: Checking radiology reports")
    print("=" * 70)

    # 固定 seed
    selection_path = ROOT / "subset" / "patient_selection.csv"
    if selection_path.exists():
        # Reproduce the submitted patient list instead of drawing a different dataset.
        frozen = pd.read_csv(selection_path)
        if len(frozen) != N_PATIENTS or frozen["subject_id"].nunique() != N_PATIENTS:
            raise ValueError("Frozen patient selection must contain 5000 unique patients")
        candidates = frozen.merge(
            candidates, on=["subject_id", "study_id", "dicom_id"],
            how="left", sort=False, validate="one_to_one", indicator=True,
        )
        if not candidates["_merge"].eq("both").all():
            raise ValueError("Frozen patient selection does not satisfy the strict AP/PA study rule")
        candidates = candidates.drop(columns="_merge")
    else:
        candidates = candidates.sample(frac=1, random_state=SEED).reset_index(drop=True)

    records = []
    missing_report_count = 0

    # name=None：
    # 返回普通 tuple
    # 后面用 candidates.columns 恢复原始列名
    pbar = tqdm(
        candidates.itertuples(
            index=False,
            name=None
        ),
        total=len(candidates),
        desc="Selecting valid patients",
        unit="patient",
        disable=not sys.stderr.isatty(),
    )

    for row_values in pbar:

        # 使用原始列名：
        # 使用原始 DataFrame 列名重新构造字典
        # 这样 "Pleural Effusion" 不会被重命名
        row = dict(
            zip(
                candidates.columns,
                row_values
            )
        )

        subject_id = int(
            row["subject_id"]
        )

        study_id = int(
            row["study_id"]
        )

        report = load_report(
            subject_id,
            study_id,
        )

        if report is None:

            missing_report_count += 1

            pbar.set_postfix(
                refresh=False,
                valid=len(records),
                missing=missing_report_count,
                target=N_PATIENTS,
            )

            continue

        # 保留所有原始列名
        record = row.copy()

        # 添加处理后的报告
        record["report"] = report

        records.append(record)

        pbar.set_postfix(
            refresh=False,
            valid=len(records),
            missing=missing_report_count,
            target=N_PATIENTS,
        )

        # 找够 5000 人立即停止
        if len(records) >= N_PATIENTS:
            break


    pbar.close()


    if len(records) < N_PATIENTS:

        raise RuntimeError(
            f"Only {len(records)} valid patients were found. "
            f"{N_PATIENTS} patients are required."
        )


    selected = pd.DataFrame(
        records
    ).reset_index(drop=True)


    print("\nPatient selection finished.")

    print(
        f"Selected patients: "
        f"{selected['subject_id'].nunique():,}"
    )

    print(
        f"Selected studies: "
        f"{selected['study_id'].nunique():,}"
    )

    print(
        f"Selected images: "
        f"{len(selected):,}"
    )

    print(
        f"Rejected due to missing Findings/Impression: "
        f"{missing_report_count:,}"
    )


    # 额外检查疾病标签列是否都保留
    print("\nDisease columns check:")

    for disease in DISEASES:

        if disease in selected.columns:
            print(f"[OK] {disease}")

        else:
            print(f"[MISSING] {disease}")


    # ============================================================
    # 8. 患者级 train / val / test = 8 : 1 : 1
    # ============================================================

    print("\n" + "=" * 70)
    print("Step 8: Patient-level train / val / test split")
    print("=" * 70)

    patient_ids = (
        selected["subject_id"]
        .drop_duplicates()
        .to_numpy()
        .copy()
    )

    if len(patient_ids) != N_PATIENTS:
        raise RuntimeError(
            "Patient count is not equal to 5000."
        )

    rng = np.random.default_rng(SEED)

    rng.shuffle(patient_ids)

    n_train = int(
        N_PATIENTS * 0.8
    )

    n_val = int(
        N_PATIENTS * 0.1
    )

    train_ids = set(
        patient_ids[
            :n_train
        ].tolist()
    )

    val_ids = set(
        patient_ids[
            n_train:
            n_train + n_val
        ].tolist()
    )

    test_ids = set(
        patient_ids[
            n_train + n_val:
        ].tolist()
    )


    def assign_split(subject_id):

        subject_id = int(subject_id)

        if subject_id in train_ids:
            return "train"

        if subject_id in val_ids:
            return "val"

        if subject_id in test_ids:
            return "test"

        raise RuntimeError(
            f"Unknown subject_id: {subject_id}"
        )


    selected["split"] = selected[
        "subject_id"
    ].apply(assign_split)

    split_counts = (
        selected["split"]
        .value_counts()
        .reindex(
            ["train", "val", "test"]
        )
    )

    print(split_counts)

    assert split_counts["train"] == 4000
    assert split_counts["val"] == 500
    assert split_counts["test"] == 500


    # ============================================================
    # 9. 生成 labels 和 mask
    # ============================================================

    print("\n" + "=" * 70)
    print("Step 9: Creating labels and masks")
    print("=" * 70)

    """
    指导书规定：

    label = 0 或 1:
        有效
        mask = 1

    label = -1 或 NaN:
        无效
        mask = 0

    这里无效标签在 labels 中统一填 0。
    真正是否参与后续计算由 mask 控制。
    """


    def make_labels_and_mask(row):

        labels = []
        mask = []

        for disease in DISEASES:

            value = row[disease]

            if pd.notna(value) and value in [0, 1]:

                labels.append(
                    int(value)
                )

                mask.append(1)

            else:

                labels.append(0)

                mask.append(0)

        return pd.Series(
            {
                "labels": json.dumps(labels),
                "mask": json.dumps(mask),
            }
        )


    label_mask_df = selected.apply(
        make_labels_and_mask,
        axis=1,
    )

    selected["labels"] = label_mask_df["labels"]
    selected["mask"] = label_mask_df["mask"]

    print("Labels and masks created.")


    # ============================================================
    # 10. 生成 JPG 相对路径和下载 URL
    # ============================================================

    print("\n" + "=" * 70)
    print("Step 10: Generating JPG download URLs")
    print("=" * 70)


    def create_image_information(row):

        subject_id = str(
            int(row["subject_id"])
        )

        study_id = str(
            int(row["study_id"])
        )

        dicom_id = str(
            row["dicom_id"]
        )

        prefix = (
            "p"
            + subject_id[:2]
        )

        relative_path = (
            f"{prefix}/"
            f"p{subject_id}/"
            f"s{study_id}/"
            f"{dicom_id}.jpg"
        )

        image_path = (
            "images/"
            + relative_path
        )

        download_url = (
            f"{BASE_URL}/"
            f"{relative_path}"
        )

        return pd.Series(
            {
                "relative_path": relative_path,
                "image_path": image_path,
                "download_url": download_url,
            }
        )


    image_info = selected.apply(
        create_image_information,
        axis=1,
    )

    selected[
        [
            "relative_path",
            "image_path",
            "download_url",
        ]
    ] = image_info[
        [
            "relative_path",
            "image_path",
            "download_url",
        ]
    ]

    print(
        f"Generated URLs: {len(selected):,}"
    )


    # ============================================================
    # 11. patient_id
    # ============================================================

    selected["patient_id"] = selected[
        "subject_id"
    ]


    # ============================================================
    # 12. 保存项目要求的主索引
    # ============================================================

    print("\n" + "=" * 70)
    print("Step 11: Saving dataset manifest")
    print("=" * 70)

    manifest_columns = [
        "patient_id",
        "study_id",
        "image_path",
        "report",
        "split",
        "labels",
        "mask",
    ]

    manifest = selected[
        manifest_columns
    ].copy()

    manifest_path = (
        OUT_DIR
        / "mimic_5000.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8",
    )

    print(
        f"Saved:\n{manifest_path}"
    )


    # ============================================================
    # 13. 保存带完整 ID 的调试索引
    # ============================================================

    debug_columns = [
        "subject_id",
        "study_id",
        "dicom_id",
        "ViewPosition",
        "relative_path",
        "image_path",
        "download_url",
        "report",
        "split",
    ] + DISEASES + [
        "labels",
        "mask",
    ]

    debug_path = (
        OUT_DIR
        / "mimic_5000_full.csv"
    )

    selected[
        debug_columns
    ].to_csv(
        debug_path,
        index=False,
        encoding="utf-8",
    )

    print(
        f"Saved:\n{debug_path}"
    )


    # ============================================================
    # 15. 保存标签名称
    # ============================================================

    label_names_path = (
        OUT_DIR
        / "label_names.json"
    )

    with open(
        label_names_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            DISEASES,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Saved:\n{label_names_path}"
    )


    # ============================================================
    # 16. 统计训练集 5 个疾病的阳性样本数
    # ============================================================

    print("\n" + "=" * 70)
    print("Step 12: Training-set disease statistics")
    print("=" * 70)

    train_df = selected[
        selected["split"] == "train"
    ].copy()

    class_stats = []

    for disease in DISEASES:

        values = train_df[disease]

        valid_mask = (
            values.isin([0, 1])
        )

        positive_count = int(
            (values == 1).sum()
        )

        negative_count = int(
            (values == 0).sum()
        )

        valid_count = int(
            valid_mask.sum()
        )

        uncertain_or_missing = int(
            len(values)
            - valid_count
        )

        class_stats.append(
            {
                "disease": disease,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "valid_count": valid_count,
                "uncertain_or_missing": uncertain_or_missing,
            }
        )

    class_stats_df = pd.DataFrame(
        class_stats
    )

    class_stats_df = class_stats_df.sort_values(
        "positive_count",
        ascending=False,
    ).reset_index(drop=True)

    class_stats_path = (
        OUT_DIR
        / "train_class_statistics.csv"
    )

    class_stats_df.to_csv(
        class_stats_path,
        index=False,
    )

    print(class_stats_df)

    print(
        f"\nSaved:\n{class_stats_path}"
    )


    # ============================================================
    # 17. 高频组和低频组
    # ============================================================

    print("\n" + "=" * 70)
    print("Step 13: Creating frequency groups")
    print("=" * 70)

    sorted_diseases = class_stats_df[
        "disease"
    ].tolist()

    high_frequency = (
        sorted_diseases[:2]
    )

    low_frequency = (
        sorted_diseases[-2:]
    )

    middle_frequency = (
        sorted_diseases[2:-2]
    )

    frequency_groups = {
        "high_frequency": high_frequency,
        "middle_frequency": middle_frequency,
        "low_frequency": low_frequency,
    }

    frequency_path = (
        OUT_DIR
        / "frequency_groups.json"
    )

    with open(
        frequency_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            frequency_groups,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "High-frequency diseases:",
        high_frequency,
    )

    print(
        "Low-frequency diseases:",
        low_frequency,
    )

    print(
        f"Saved:\n{frequency_path}"
    )


    # ============================================================
    # 18. 最终一致性检查
    # ============================================================

    print("\n" + "=" * 70)
    print("Step 14: Final consistency checks")
    print("=" * 70)

    assert len(selected) == 5000

    assert (
        selected["subject_id"].nunique()
        == 5000
    )

    assert (
        selected["split"].value_counts()["train"]
        == 4000
    )

    assert (
        selected["split"].value_counts()["val"]
        == 500
    )

    assert (
        selected["split"].value_counts()["test"]
        == 500
    )

    assert (
        selected["report"]
        .notna()
        .all()
    )

    assert (
        selected["image_path"]
        .notna()
        .all()
    )

    assert (
        selected["download_url"]
        .notna()
        .all()
    )

    duplicate_patients = (
        selected["subject_id"]
        .duplicated()
        .sum()
    )

    if duplicate_patients != 0:
        raise RuntimeError(
            f"Duplicate patients found: "
            f"{duplicate_patients}"
        )

    print("All consistency checks passed.")


    # ============================================================
    # 19. 输出总结
    # ============================================================

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETED")
    print("=" * 70)

    print(
        f"""
    Total patients : {len(selected):,}

    Train          : {split_counts['train']:,}
    Validation     : {split_counts['val']:,}
    Test           : {split_counts['test']:,}

    Output directory:
    {OUT_DIR}

    Main manifest:
    {manifest_path}

    Full manifest:
    {debug_path}

    Class statistics:
    {class_stats_path}

    Frequency groups:
    {frequency_path}
    """
    )

    selection_path = OUT_DIR / "patient_selection.csv"
    selected[["subject_id", "study_id", "dicom_id"]].to_csv(selection_path, index=False, encoding="utf-8")
    with open(OUT_DIR / "selection_provenance.json", "w", encoding="utf-8") as f:
        json.dump({
            "policy": "fixed_subset_v1", "seed": SEED,
            "patient_selection_file": selection_path.relative_to(ROOT).as_posix(),
            "patient_selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            "selection_rule": "Smallest study_id across all studies, then AP/PA, then smallest dicom_id; Findings preferred, Impression fallback.",
            "patient_count": N_PATIENTS,
            "split_counts": {"train": 4000, "val": 500, "test": 500},
        }, f, ensure_ascii=False, indent=2)

    print("Next step: run download_images.py using mimic_5000_full.csv.")

if __name__ == "__main__":
    main()
