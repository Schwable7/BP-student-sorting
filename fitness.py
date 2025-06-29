import logging
import numpy as np

from constants import GENDER, MALE, FEMALE, DEFERRAL, LEARNING_DISABILITIES, TALENT, DIFF_MOTHER_LANG, TOGETHER


def fitness_simple(classes: list[list], print_progress: bool = False) -> dict[str, float]:
    # ------------------------------------------------------------------
    # 1. Pass once through all students – count attributes & build maps
    # ------------------------------------------------------------------
    total_classes = len(classes)

    # per-class counts
    class_sizes = np.zeros(total_classes, dtype=int)
    boys_counts = np.zeros(total_classes, dtype=int)
    girls_counts = np.zeros(total_classes, dtype=int)
    deferred_counts = np.zeros(total_classes, dtype=int)
    disabilities_counts = np.zeros(total_classes, dtype=int)
    talent_counts = np.zeros(total_classes, dtype=int)
    lang_counts = np.zeros(total_classes, dtype=int)

    # totals
    total_boys = total_girls = total_deferred = 0
    total_disabilities = total_talent = total_lang = 0

    # mapping id(student) → class_idx
    stu2cls: dict[int, int] = {}

    for cls_idx, cls in enumerate(classes):
        class_sizes[cls_idx] = len(cls)
        for s in cls:
            sid = id(s)
            stu2cls[sid] = cls_idx

            # gender
            if s[GENDER] == MALE:
                boys_counts[cls_idx] += 1
                total_boys += 1
            elif s[GENDER] == FEMALE:
                girls_counts[cls_idx] += 1
                total_girls += 1

            # other binary attributes
            if s[DEFERRAL] == 1:
                deferred_counts[cls_idx] += 1
                total_deferred += 1
            if s[LEARNING_DISABILITIES] == 1:
                disabilities_counts[cls_idx] += 1
                total_disabilities += 1
            if s[TALENT] == 1:
                talent_counts[cls_idx] += 1
                total_talent += 1
            if s[DIFF_MOTHER_LANG] == 1:
                lang_counts[cls_idx] += 1
                total_lang += 1

    total_students = class_sizes.sum()  # cheaper than len(all_students)

    # ------------------------------------------------------------------
    # 2. Dispersion metrics using CV²
    # ------------------------------------------------------------------
    size_dev = normalized_mse(class_sizes, total_students)
    boys_dev = normalized_mse(boys_counts, total_boys)
    girls_dev = normalized_mse(girls_counts, total_girls)
    gender_dev = boys_dev + girls_dev
    deferred_dev = normalized_mse(deferred_counts, total_deferred)
    disabilities_dev = normalized_mse(disabilities_counts, total_disabilities)
    talent_dev = normalized_mse(talent_counts, total_talent)
    lang_dev = normalized_mse(lang_counts, total_lang)


    # ------------------------------------------------------------------
    # 5. Weighted cost (weights unchanged, except small tweak you had)
    # ------------------------------------------------------------------
    WEIGHTS = {
        "size_dev": 1.0,
        "gender_dev": 1.0,
        "deferred_dev": 1.0,
        "disabilities_dev": 1.0,
        "talent_dev": 1.0,
        "diff_lang_dev": 1.0,
        "together_penalty": 0.05,
        "not_together_penalty": 0.05,
    }

    weighted_size_dev = WEIGHTS["size_dev"] * size_dev
    weighted_gender_dev = WEIGHTS["gender_dev"] * gender_dev
    weighted_deferred_dev = WEIGHTS["deferred_dev"] * deferred_dev
    weighted_disabilities_dev = WEIGHTS["disabilities_dev"] * disabilities_dev
    weighted_talented_dev = WEIGHTS["talent_dev"] * talent_dev
    weighted_diff_lang_dev = WEIGHTS["diff_lang_dev"] * lang_dev

    total_cost = (
            weighted_size_dev +
            weighted_gender_dev +
            weighted_deferred_dev +
            weighted_disabilities_dev +
            weighted_talented_dev +
            weighted_diff_lang_dev
    )

    if print_progress:
        logging.info(
            "Fitness evaluation -> "
            f"Size Dev: {size_dev:.4f}, Gender Dev: {gender_dev:.4f}, Deferred: {deferred_dev:.4f}, "
            f"Disabilities: {disabilities_dev:.4f}, Talent: {talent_dev:.4f}, Lang: {lang_dev:.4f}, "
            f"Total: {total_cost:.4f}"
        )

    return {
        "total_cost": total_cost,
        "size_dev": float(size_dev),
        "boys_dev": float(boys_dev),
        "girls_dev": float(girls_dev),
        "deferred_dev": float(deferred_dev),
        "disabilities_dev": float(disabilities_dev),
        "talent_dev": float(talent_dev),
        "diff_lang_dev": float(lang_dev),
    }


def fitness_1(classes: list[list], print_progress: bool = False) -> dict[str, float]:
    # Count all students
    all_students = [s for cls in classes for s in cls]
    total_classes = len(classes)
    total_students = len(all_students)

    # Attribute totals
    total_boys = sum(1 for s in all_students if s[GENDER] == MALE)
    total_girls = sum(1 for s in all_students if s[GENDER] == FEMALE)
    total_deferred = sum(1 for s in all_students if s[DEFERRAL] == 1)
    total_disabilities = sum(1 for s in all_students if s[LEARNING_DISABILITIES] == 1)
    total_talent = sum(1 for s in all_students if s[TALENT] == 1)
    total_lang = sum(1 for s in all_students if s[DIFF_MOTHER_LANG] == 1)

    # Helper to compute normalized MSE
    def normalized_mse(values: list[int], max_range: int) -> float:
        mean = np.mean(values)
        mse = np.mean([(v - mean) ** 2 for v in values])
        return mse / (max_range ** 2) if max_range > 0 else 0.0

    def cv2(values: list[int]) -> float:
        """
        Scale-free dispersion measure: variance / mean².
        0   → perfectly even across classes
        > 0 → imbalance (unbounded above)
        """
        mean = np.mean(values)
        if mean == 0:  # attribute absent ⇒ no imbalance pressure
            return 0.0
        var = np.var(values)  # population variance (ddof=0)
        return var / (mean ** 2)

    # Attribute counts per class
    class_sizes = [len(cls) for cls in classes]
    boys_counts = [sum(1 for s in cls if s[GENDER] == MALE) for cls in classes]
    girls_counts = [sum(1 for s in cls if s[GENDER] == FEMALE) for cls in classes]
    deferred_counts = [sum(1 for s in cls if s[DEFERRAL] == 1) for cls in classes]
    disabilities_counts = [sum(1 for s in cls if s[LEARNING_DISABILITIES] == 1) for cls in classes]
    talent_counts = [sum(1 for s in cls if s[TALENT] == 1) for cls in classes]
    lang_counts = [sum(1 for s in cls if s[DIFF_MOTHER_LANG] == 1) for cls in classes]

    # Normalized MSE values
    size_dev = normalized_mse(class_sizes, total_students)
    boys_dev = normalized_mse(boys_counts, total_boys)
    girls_dev = normalized_mse(girls_counts, total_girls)
    gender_dev = boys_dev + girls_dev
    deferred_dev = normalized_mse(deferred_counts, total_deferred)
    disabilities_dev = normalized_mse(disabilities_counts, total_disabilities)
    talent_dev = normalized_mse(talent_counts, total_talent)
    lang_dev = normalized_mse(lang_counts, total_lang)

    # TOGETHER penalty
    together_penalty = 0.0
    together_students = [s for s in all_students if s[TOGETHER] == 1]
    for i, s1 in enumerate(together_students):
        for s2 in together_students[i + 1:]:
            if not any(s1 in cls and s2 in cls for cls in classes):
                together_penalty += 1.0

    max_together_violations = len(together_students) * (len(together_students) - 1) / 2
    normalized_together_penalty = together_penalty / max_together_violations if max_together_violations > 0 else 0.0

    # NOT_TOGETHER penalty
    not_together_penalty = 0.0
    not_together_keys = [key for key in all_students[0].keys() if key.startswith("not_together_")]
    for cls in classes:
        for i in range(len(cls)):
            for j in range(i + 1, len(cls)):
                s1, s2 = cls[i], cls[j]
                for key in not_together_keys:
                    if s1.get(key) == "1" and s2.get(key) == "1":
                        not_together_penalty += 1.0

    # Compute max possible NOT_TOGETHER violations
    max_not_together_violations = 0.0
    for key in not_together_keys:
        group = [s for s in all_students if s.get(key) == "1"]
        n = len(group)
        max_not_together_violations += n * (n - 1) / 2

    normalized_not_together_penalty = not_together_penalty / max_not_together_violations if max_not_together_violations > 0 else 0.0

    # Weights
    WEIGHTS = {
        "size_dev": 1.0,
        "gender_dev": 1.0,
        "deferred_dev": 1.0,
        "disabilities_dev": 1.0,
        "talent_dev": 1.0,
        "diff_lang_dev": 1.0,
        "together_penalty": 0.001,
        "not_together_penalty": 0.005,
    }

    # Weighted total cost
    weighted_size_dev = WEIGHTS["size_dev"] * size_dev
    weighted_gender_dev = WEIGHTS["gender_dev"] * gender_dev
    weighted_deferred_dev = WEIGHTS["deferred_dev"] * deferred_dev
    weighted_disabilities_dev = WEIGHTS["disabilities_dev"] * disabilities_dev
    weighted_talented_dev = WEIGHTS["talent_dev"] * talent_dev
    weighted_diff_lang_dev = WEIGHTS["diff_lang_dev"] * lang_dev
    weighted_together_penalty = WEIGHTS["together_penalty"] * normalized_together_penalty
    weighted_not_together_penalty = WEIGHTS["not_together_penalty"] * normalized_not_together_penalty
    total_cost = (
            weighted_size_dev +
            weighted_gender_dev +
            weighted_deferred_dev +
            weighted_disabilities_dev +
            weighted_talented_dev +
            weighted_diff_lang_dev +
            weighted_together_penalty +
            weighted_not_together_penalty
    )

    if print_progress:
        logging.info(
            f"Fitness evaluation -> "
            f"Size Dev: {size_dev:.4f}, Gender Dev: {gender_dev:.4f}, Deferred: {deferred_dev:.4f}, "
            f"Disabilities: {disabilities_dev:.4f}, Talent: {talent_dev:.4f}, Lang: {lang_dev:.4f}, "
            f"Together: {weighted_together_penalty:.4f}, Not-Together: {weighted_not_together_penalty:.4f}, "
            f"Total: {total_cost:.4f}"
        )

    return {
        "total_cost": total_cost,
        "size_dev": size_dev,
        "boys_dev": boys_dev,
        "girls_dev": girls_dev,
        "deferred_dev": deferred_dev,
        "disabilities_dev": disabilities_dev,
        "talent_dev": talent_dev,
        "diff_lang_dev": lang_dev,
        "together_penalty": weighted_together_penalty,
        "not_together_penalty": weighted_not_together_penalty,
    }


def cv2(values: list[int]) -> float:
    """Coefficient-of-variation squared (scale-free imbalance)."""
    mean = np.mean(values)
    if mean == 0:                # attribute absent ⇒ no pressure
        return 0.0
    return np.var(values, ddof=0) / (mean * mean)


def normalized_mse(values: list[int], max_range: int) -> float:
    mean = np.mean(values)
    mse = np.mean([(v - mean) ** 2 for v in values])
    return float(mse) #/ (max_range ** 2) if max_range > 0 else 0.0


def fitness(classes: list[list[dict]], print_progress: bool = False) -> dict[str, float]:
    # ------------------------------------------------------------------
    # 1. Pass once through all students – count attributes & build maps
    # ------------------------------------------------------------------
    total_classes = len(classes)

    # per-class counts
    class_sizes          = np.zeros(total_classes, dtype=int)
    boys_counts          = np.zeros(total_classes, dtype=int)
    girls_counts         = np.zeros(total_classes, dtype=int)
    deferred_counts      = np.zeros(total_classes, dtype=int)
    disabilities_counts  = np.zeros(total_classes, dtype=int)
    talent_counts        = np.zeros(total_classes, dtype=int)
    lang_counts          = np.zeros(total_classes, dtype=int)
    # together_ids: list[int] = []

    # totals
    total_boys = total_girls = total_deferred = 0
    total_disabilities = total_talent = total_lang = 0

    # discover “not_together_*” keys once
    first_student = classes[0][0] if classes and classes[0] else {}
    not_together_keys = [k for k in first_student.keys() if k.startswith("not_together_")]
    together_keys = [k for k in first_student.keys() if k.startswith("together_")]

    not_together_groups = {k: [] for k in not_together_keys}          # key → list[id(student)]
    together_groups = {k: [] for k in together_keys}

    # mapping id(student) → class_idx
    stu2cls: dict[int, int] = {}

    for cls_idx, cls in enumerate(classes):
        class_sizes[cls_idx] = len(cls)
        for s in cls:
            sid = id(s)
            stu2cls[sid] = cls_idx

            # gender
            if s[GENDER] == MALE:
                boys_counts[cls_idx]  += 1
                total_boys           += 1
            elif s[GENDER] == FEMALE:
                girls_counts[cls_idx] += 1
                total_girls          += 1

            # other binary attributes
            if s[DEFERRAL] == 1:
                deferred_counts[cls_idx] += 1
                total_deferred           += 1
            if s[LEARNING_DISABILITIES] == 1:
                disabilities_counts[cls_idx] += 1
                total_disabilities           += 1
            if s[TALENT] == 1:
                talent_counts[cls_idx] += 1
                total_talent           += 1
            if s[DIFF_MOTHER_LANG] == 1:
                lang_counts[cls_idx] += 1
                total_lang           += 1

            # “must be together”
            for k in together_keys:
                if s.get(k) == "1":
                    together_groups[k].append(sid)

            # “must not be together” groups
            for k in not_together_keys:
                if s.get(k) == "1":
                    not_together_groups[k].append(sid)

    total_students = class_sizes.sum()   # cheaper than len(all_students)

    # ------------------------------------------------------------------
    # 2. Dispersion metrics using CV²
    # ------------------------------------------------------------------
    size_dev = normalized_mse(class_sizes, total_students)
    boys_dev = normalized_mse(boys_counts, total_boys)
    girls_dev = normalized_mse(girls_counts, total_girls)
    gender_dev = boys_dev + girls_dev
    deferred_dev = normalized_mse(deferred_counts, total_deferred)
    disabilities_dev = normalized_mse(disabilities_counts, total_disabilities)
    talent_dev = normalized_mse(talent_counts, total_talent)
    lang_dev = normalized_mse(lang_counts, total_lang)

    # ------------------------------------------------------------------
    # 2. Dispersion metrics using MSE
    # ------------------------------------------------------------------
    # size_dev = cv2(class_sizes)
    # boys_dev = cv2(boys_counts)
    # girls_dev = cv2(girls_counts)
    # gender_dev = boys_dev + girls_dev
    # deferred_dev = cv2(deferred_counts)
    # disabilities_dev = cv2(disabilities_counts)
    # talent_dev = cv2(talent_counts)
    # lang_dev = cv2(lang_counts)

    # ------------------------------------------------------------------
    # 3. TOGETHER penalty – pairs in different classes
    # ------------------------------------------------------------------

    together_penalty = 0
    max_together_violations = 0
    together_group_penalties = []

    for key, ids in together_groups.items():
        ids = np.array(ids, dtype=int)
        n = len(ids)
        if n < 2:
            continue
        per_cls = np.bincount([stu2cls[sid] for sid in ids], minlength=total_classes)
        pairs_within = np.sum(per_cls * (per_cls - 1) // 2)
        total_pairs = n * (n - 1) // 2
        violations = total_pairs - pairs_within
        together_group_penalties.append(violations)

    # MSE instead of normalized ratio
    together_penalty_mse = normalized_mse(together_group_penalties, max_range=1)  # no normalization

    # ------------------------------------------------------------------
    # 4. NOT_TOGETHER penalty – pairs that share a class
    # ------------------------------------------------------------------
    not_together_penalty = 0
    max_not_together_violations = 0

    not_together_group_penalties = []

    for key, ids in not_together_groups.items():
        ids = np.array(ids, dtype=int)
        n = len(ids)
        if n < 2:
            continue
        per_cls = np.bincount([stu2cls[sid] for sid in ids], minlength=total_classes)
        pairs_within = np.sum(per_cls * (per_cls - 1) // 2)
        total_pairs = n * (n - 1) // 2
        violations = total_pairs - pairs_within
        not_together_group_penalties.append(violations)

    not_together_penalty_mse = normalized_mse(not_together_group_penalties, max_range=1)  # again, no normalization

    # ------------------------------------------------------------------
    # 5. Weighted cost (weights unchanged, except small tweak you had)
    # ------------------------------------------------------------------
    WEIGHTS = {
        "size_dev": 1.0,
        "gender_dev": 1.0,
        "deferred_dev": 1.0,
        "disabilities_dev": 1.0,
        "talent_dev": 1.0,
        "diff_lang_dev": 1.0,

        # basic dataset
        "together_penalty": 0.05,
        "not_together_penalty": 0.05,

        # skewed dataset
        # "together_penalty": 0.01,
        # "not_together_penalty": 0.01,

        # large dataset
        # "together_penalty": 0.005,
        # "not_together_penalty": 0.005,
    }

    weighted_size_dev            = WEIGHTS["size_dev"] * size_dev
    weighted_gender_dev          = WEIGHTS["gender_dev"] * gender_dev
    weighted_deferred_dev        = WEIGHTS["deferred_dev"] * deferred_dev
    weighted_disabilities_dev    = WEIGHTS["disabilities_dev"] * disabilities_dev
    weighted_talented_dev        = WEIGHTS["talent_dev"] * talent_dev
    weighted_diff_lang_dev       = WEIGHTS["diff_lang_dev"] * lang_dev
    weighted_together_penalty    = WEIGHTS["together_penalty"] * together_penalty_mse
    weighted_not_together_penalty = WEIGHTS["not_together_penalty"] * not_together_penalty_mse

    total_cost = (
        weighted_size_dev +
        weighted_gender_dev +
        weighted_deferred_dev +
        weighted_disabilities_dev +
        weighted_talented_dev +
        weighted_diff_lang_dev +
        weighted_together_penalty +
        weighted_not_together_penalty
    )

    if print_progress:
        logging.info(
            "Fitness evaluation -> "
            f"Size Dev: {size_dev:.4f}, Gender Dev: {gender_dev:.4f}, Deferred: {deferred_dev:.4f}, "
            f"Disabilities: {disabilities_dev:.4f}, Talent: {talent_dev:.4f}, Lang: {lang_dev:.4f}, "
            f"Together: {weighted_together_penalty:.4f}, Not-Together: {weighted_not_together_penalty:.4f}, "
            f"Total: {total_cost:.4f}"
        )

    return {
        "total_cost": total_cost,
        "size_dev": float(size_dev),
        "boys_dev": float(boys_dev),
        "girls_dev": float(girls_dev),
        "deferred_dev": float(deferred_dev),
        "disabilities_dev": float(disabilities_dev),
        "talent_dev": float(talent_dev),
        "diff_lang_dev": float(lang_dev),
        "together_penalty": float(weighted_together_penalty),
        "not_together_penalty": float(weighted_not_together_penalty),
    }
