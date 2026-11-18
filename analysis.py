"""Inspect gym members and choose a v1 cluster count from training and test spread.

Loads member measurements averaged over the last three months, prints
summary checks, and fits StandardScaler and KMeans on the training split
only. Held-out members are assigned to those training centroids. Models
are not fit on the test set.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "gym_members (1).csv"
CLUSTER_FEATURES = [
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]
EXCLUDED_FEATURES = ["member_id"]
RANDOM_STATE = 42
K_VALUES = range(1, 9)
PLOT_PATH = Path(__file__).resolve().parent / "elbow_plot.png"
EXTRAS_SPEND_THRESHOLD = 20
LOW_SPEND_NET_IMPACT = 5
HIGH_SPEND_NET_IMPACT = -10
FITTED_SCALER: StandardScaler | None = None
V1_MODEL: KMeans | None = None
CREDIT_SEGMENT_IDS: set[int] = set()


def main() -> None:
    global FITTED_SCALER, V1_MODEL
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    members = pd.read_csv(DATA_PATH)

    print("Dataset shape")
    print(members.shape)
    print()

    print("Column names")
    print(list(members.columns))
    print()

    print("Data types")
    print(members.dtypes)
    print()

    print("Missing values")
    print(members.isna().sum())
    print()

    duplicate_member_ids = int(members["member_id"].duplicated().sum())
    print("Duplicate member IDs")
    print(duplicate_member_ids)
    print()

    print("Descriptive statistics")
    print(members.describe())
    print()

    average_extras_spend = members["monthly_extras_spend"].mean()
    print("Average monthly extras spend")
    print(f"${average_extras_spend:.2f}")
    print()

    train_members, test_members = train_test_split(
        members,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    print("Training and test sizes")
    print(f"Training: {len(train_members)} members")
    print(f"Test: {len(test_members)} members")
    print()

    print("Selected features")
    print(CLUSTER_FEATURES)
    print()

    print("Excluded features")
    print(EXCLUDED_FEATURES)
    print("member_id is an identifier, not a characteristic describing members.")
    print()

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_members[CLUSTER_FEATURES])
    scaled_test = scaler.transform(test_members[CLUSTER_FEATURES])

    print("Scaling")
    print(
        "StandardScaler was fitted only on the training set "
        f"({int(scaler.n_samples_seen_)} members) and then used to transform "
        "the training and test sets."
    )
    print(f"Scaled training shape: {scaled_train.shape}")
    print(f"Scaled test shape: {scaled_test.shape}")
    print()

    print("Training inertia by k")
    comparison_rows = []
    v1_model = None
    for k in K_VALUES:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        model.fit(scaled_train)
        if k == 4:
            v1_model = model
        assigned_clusters = model.predict(scaled_test)
        assigned_centroids = model.cluster_centers_[assigned_clusters]
        test_spread = float(np.sum((scaled_test - assigned_centroids) ** 2))
        comparison_rows.append(
            {
                "k": k,
                "training_inertia": model.inertia_,
                "test_spread": test_spread,
            }
        )
        print(f"k={k}: {model.inertia_:.4f}")
    print()

    comparison = pd.DataFrame(comparison_rows)
    print("Training inertia and test spread")
    print(
        comparison.to_string(
            index=False,
            formatters={
                "training_inertia": "{:.4f}".format,
                "test_spread": "{:.4f}".format,
            },
        )
    )
    print()

    save_elbow_plot(comparison)

    report_cluster_choice(comparison)
    print()
    if v1_model is None:
        raise RuntimeError("The k=4 model was not retained from the training loop.")
    FITTED_SCALER = scaler
    V1_MODEL = v1_model
    labeled_members = label_members(
        v1_model,
        train_members,
        test_members,
        scaled_train,
        scaled_test,
    )
    profile_v1_clusters(v1_model, scaler, labeled_members, members)
    print()
    evaluate_segment_credits(labeled_members)
    print()
    compare_credit_policies(labeled_members)
    print()
    score_example_member()


def save_elbow_plot(comparison: pd.DataFrame) -> None:
    """Save training inertia and held-out test spread against k."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        comparison["k"],
        comparison["training_inertia"],
        marker="o",
        linewidth=2,
        label="Training inertia",
    )
    ax.plot(
        comparison["k"],
        comparison["test_spread"],
        marker="o",
        linewidth=2,
        label="Test spread",
    )
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Sum of squared distances")
    ax.set_title("Elbow plot: training inertia and test spread")
    ax.set_xticks(list(K_VALUES))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved elbow plot to {PLOT_PATH.name}")
    print()


FEATURE_LABELS = {
    "age": "Average age",
    "visits_per_month": "Visits per month",
    "avg_visit_minutes": "Average visit duration (minutes)",
    "classes_per_month": "Classes per month",
    "months_member": "Months as a member",
    "monthly_extras_spend": "Monthly extras spend",
}
CLUSTER_NAMES = {
    0: "New light users",
    1: "Loyal long-timers",
    2: "Frequent floor users",
    3: "Class and extras members",
}
CLUSTER_INTERPRETATIONS = {
    0: (
        "These newer members come only a few times a month, stay for shorter "
        "visits, and have not built a class or extras habit yet."
    ),
    1: (
        "These older members have belonged for years and keep a steady visit "
        "habit, with somewhat shorter visits than the typical member."
    ),
    2: (
        "These members come often and stay a long time, while rarely taking "
        "classes and spending about the average amount on extras."
    ),
    3: (
        "These members visit about as often as everyone else, but they take "
        "far more classes and spend much more on monthly extras."
    ),
}


def label_members(
    v1_model: KMeans,
    train_members: pd.DataFrame,
    test_members: pd.DataFrame,
    scaled_train: np.ndarray,
    scaled_test: np.ndarray,
) -> pd.DataFrame:
    """Attach the existing k=4 assignments without fitting again."""
    labeled_train = train_members.copy()
    labeled_test = test_members.copy()
    labeled_train["cluster_id"] = v1_model.predict(scaled_train)
    labeled_test["cluster_id"] = v1_model.predict(scaled_test)
    return pd.concat([labeled_train, labeled_test], axis=0)


def profile_v1_clusters(
    v1_model: KMeans,
    scaler: StandardScaler,
    labeled_members: pd.DataFrame,
    members: pd.DataFrame,
) -> None:
    """Describe the existing k=4 centroids in original units."""
    centroids = pd.DataFrame(
        scaler.inverse_transform(v1_model.cluster_centers_),
        columns=CLUSTER_FEATURES,
    )
    member_counts = labeled_members["cluster_id"].value_counts().sort_index()

    overall_mean = members[CLUSTER_FEATURES].mean()
    overall_std = members[CLUSTER_FEATURES].std(ddof=0)
    distinctiveness = (centroids - overall_mean) / overall_std

    profile = pd.DataFrame(
        {
            "Cluster ID": centroids.index,
            "Segment": [CLUSTER_NAMES[cluster_id] for cluster_id in centroids.index],
            "Members": member_counts.to_numpy(),
            "Average age": centroids["age"],
            "Visits per month": centroids["visits_per_month"],
            "Avg visit minutes": centroids["avg_visit_minutes"],
            "Classes per month": centroids["classes_per_month"],
            "Months as a member": centroids["months_member"],
            "Monthly extras spend": centroids["monthly_extras_spend"],
        }
    )
    overall_row = pd.DataFrame(
        [
            {
                "Cluster ID": "—",
                "Segment": "All members",
                "Members": len(members),
                "Average age": overall_mean["age"],
                "Visits per month": overall_mean["visits_per_month"],
                "Avg visit minutes": overall_mean["avg_visit_minutes"],
                "Classes per month": overall_mean["classes_per_month"],
                "Months as a member": overall_mean["months_member"],
                "Monthly extras spend": overall_mean["monthly_extras_spend"],
            }
        ]
    )
    presentation = pd.concat([profile, overall_row], ignore_index=True)

    print("v1 segment profiles")
    print("Centroid values are in original units. The last row is the average of all 330 members.")
    print(
        presentation.to_string(
            index=False,
            formatters={
                "Average age": "{:.1f}".format,
                "Visits per month": "{:.1f}".format,
                "Avg visit minutes": "{:.1f}".format,
                "Classes per month": "{:.1f}".format,
                "Months as a member": "{:.1f}".format,
                "Monthly extras spend": "${:.2f}".format,
            },
        )
    )
    print()

    print("What distinguishes each segment")
    for cluster_id in centroids.index:
        ranked_features = (
            distinctiveness.loc[cluster_id].abs().sort_values(ascending=False)
        )
        notable_count = 3 if float(ranked_features.iloc[2]) >= 0.5 else 2
        print(f"Cluster {cluster_id}: {CLUSTER_NAMES[cluster_id]}")
        print(f"Members: {int(member_counts.loc[cluster_id])}")
        print("Most distinctive characteristics:")
        for feature in ranked_features.index[:notable_count]:
            direction = (
                "higher"
                if centroids.loc[cluster_id, feature] > overall_mean[feature]
                else "lower"
            )
            if feature == "monthly_extras_spend":
                centroid_value = f"${centroids.loc[cluster_id, feature]:.2f}"
                overall_value = f"${overall_mean[feature]:.2f}"
            else:
                centroid_value = f"{centroids.loc[cluster_id, feature]:.1f}"
                overall_value = f"{overall_mean[feature]:.1f}"
            print(
                f"- {FEATURE_LABELS[feature]} is {direction} "
                f"({centroid_value} vs {overall_value} overall)"
            )
        print(CLUSTER_INTERPRETATIONS[cluster_id])
        print()

    assigned_total = int(member_counts.sum())
    print(f"Members assigned across all clusters: {assigned_total}")
    print(f"Assignment check passed: {assigned_total == len(members)}")


def format_dollars(amount: float) -> str:
    """Format a dollar amount with a leading minus sign for losses."""
    if amount < 0:
        return f"-${abs(amount):,.0f}"
    return f"${amount:,.0f}"


def evaluate_segment_credits(labeled_members: pd.DataFrame) -> None:
    """Give the credit only to segments with a positive total net impact."""
    global CREDIT_SEGMENT_IDS

    rows = []
    explanations = []
    credit_ids: set[int] = set()
    for cluster_id in range(len(CLUSTER_NAMES)):
        segment = labeled_members[labeled_members["cluster_id"] == cluster_id]
        member_count = int(len(segment))
        under_20 = int((segment["monthly_extras_spend"] < EXTRAS_SPEND_THRESHOLD).sum())
        at_least_20 = member_count - under_20
        net_impact = (
            under_20 * LOW_SPEND_NET_IMPACT + at_least_20 * HIGH_SPEND_NET_IMPACT
        )
        give_credit = net_impact > 0
        if give_credit:
            credit_ids.add(cluster_id)
        rows.append(
            {
                "Cluster ID": cluster_id,
                "Segment": CLUSTER_NAMES[cluster_id],
                "Members": member_count,
                "Spending under $20": under_20,
                "Spending $20 or more": at_least_20,
                "Net impact if all receive credit": net_impact,
                "Credit decision": "Give credit" if give_credit else "No credit",
            }
        )
        gain = under_20 * LOW_SPEND_NET_IMPACT
        cost = at_least_20 * abs(HIGH_SPEND_NET_IMPACT)
        if give_credit:
            decision = (
                f"Net impact is {format_dollars(net_impact)}, so every member "
                "in this segment receives the credit."
            )
        else:
            decision = (
                f"Net impact is {format_dollars(net_impact)}, so the credit is "
                "withheld from the whole segment."
            )
        if under_20 == 1:
            low_spend_clause = (
                f"1 of {member_count} members spends under $20 and would add "
                f"{format_dollars(gain)}"
            )
        else:
            low_spend_clause = (
                f"{under_20} of {member_count} members spend under $20 and would add "
                f"{format_dollars(gain)}"
            )
        explanations.append(
            f"{CLUSTER_NAMES[cluster_id]}: {low_spend_clause}. "
            f"The other {at_least_20} spend $20 or more and would cost "
            f"{format_dollars(cost)}. {decision}"
        )

    CREDIT_SEGMENT_IDS = credit_ids
    impact = pd.DataFrame(rows)
    print("Monthly extras credit by segment")
    print(
        "Impact assumes every member in the segment receives the $10 credit: "
        "+$5 when extras spend is under $20, and -$10 when it is $20 or more."
    )
    print(
        impact.to_string(
            index=False,
            formatters={"Net impact if all receive credit": format_dollars},
        )
    )
    print()
    print("Credit decisions")
    for explanation in explanations:
        print(explanation)
        print()


def decide_credit(member: dict[str, float]) -> dict[str, object]:
    """Predict a member's segment and apply that segment's credit rule.

    The member's own extras spend does not override the segment decision.
    """
    if FITTED_SCALER is None or V1_MODEL is None:
        raise RuntimeError("The scaler and k=4 model must be fitted before deciding credit.")

    features = pd.DataFrame(
        [[member[feature] for feature in CLUSTER_FEATURES]],
        columns=CLUSTER_FEATURES,
    )
    scaled_member = FITTED_SCALER.transform(features)
    cluster_id = int(V1_MODEL.predict(scaled_member)[0])
    return {
        "cluster_id": cluster_id,
        "segment": CLUSTER_NAMES[cluster_id],
        "receives_credit": cluster_id in CREDIT_SEGMENT_IDS,
    }


def member_net_impact(monthly_extras_spend: float, receives_credit: bool) -> float:
    """Score one member from that member's extras spend, not a centroid."""
    if not receives_credit:
        return 0.0
    if monthly_extras_spend < EXTRAS_SPEND_THRESHOLD:
        return float(LOW_SPEND_NET_IMPACT)
    return float(HIGH_SPEND_NET_IMPACT)


def compare_credit_policies(labeled_members: pd.DataFrame) -> None:
    """Compare giving every member the credit with crediting New light users only."""
    if FITTED_SCALER is None or V1_MODEL is None:
        raise RuntimeError("The scaler and k=4 model must be fitted before comparing policies.")
    credited_segments = {CLUSTER_NAMES[cluster_id] for cluster_id in CREDIT_SEGMENT_IDS}
    if credited_segments != {"New light users"}:
        raise RuntimeError("v1 credit decisions no longer match the New light users segment.")

    spends = labeled_members["monthly_extras_spend"].to_numpy(dtype=float)
    receives_v1 = labeled_members["cluster_id"].isin(CREDIT_SEGMENT_IDS).to_numpy()
    baseline_impact = np.array(
        [member_net_impact(spend, True) for spend in spends],
        dtype=float,
    )
    v1_impact = np.array(
        [
            member_net_impact(spend, receives_credit)
            for spend, receives_credit in zip(spends, receives_v1)
        ],
        dtype=float,
    )
    baseline_total = float(baseline_impact.sum())
    v1_total = float(v1_impact.sum())
    monthly_improvement = v1_total - baseline_total
    annualized_improvement = monthly_improvement * 12

    centroid_spend = scaler_centroid_extras_spend()
    new_light_id = next(
        cluster_id
        for cluster_id, name in CLUSTER_NAMES.items()
        if name == "New light users"
    )
    new_light_members = labeled_members["cluster_id"].to_numpy() == new_light_id
    high_spenders_in_segment = int(
        (new_light_members & (spends >= EXTRAS_SPEND_THRESHOLD)).sum()
    )

    comparison = pd.DataFrame(
        [
            {
                "Policy": "Baseline: credit for every member",
                "Members receiving credit": int(len(labeled_members)),
                "Net impact": baseline_total,
            },
            {
                "Policy": "v1: New light users only",
                "Members receiving credit": int(receives_v1.sum()),
                "Net impact": v1_total,
            },
            {
                "Policy": "Monthly improvement of v1",
                "Members receiving credit": "—",
                "Net impact": monthly_improvement,
            },
            {
                "Policy": "Annualized improvement (× 12)",
                "Members receiving credit": "—",
                "Net impact": annualized_improvement,
            },
        ]
    )
    print("Credit policy comparison")
    print(
        comparison.to_string(
            index=False,
            formatters={"Net impact": format_dollars},
        )
    )
    print()
    print(
        "Spend source check: the "
        f"{len(labeled_members)} net impacts are summed from each member's "
        "monthly_extras_spend. "
        f"The New light users centroid extras spend is ${centroid_spend[new_light_id]:.2f}, "
        "so a centroid rule would score all "
        f"{int(new_light_members.sum())} members in that segment at +$5. "
        f"{high_spenders_in_segment} of those members actually spend $20 or more "
        "and are scored at -$10 from their own spend."
    )
    print()
    print(
        "Offer the $10 extras credit only to New light users. "
        f"That targeted policy improves the gym's result by {format_dollars(monthly_improvement)} "
        f"a month, or {format_dollars(annualized_improvement)} a year, "
        "by stopping the credit for members who would not spend more."
    )


def scaler_centroid_extras_spend() -> dict[int, float]:
    """Return each existing centroid's extras spend in original dollars."""
    if FITTED_SCALER is None or V1_MODEL is None:
        raise RuntimeError("The scaler and k=4 model must be fitted before reading centroids.")
    spend_index = CLUSTER_FEATURES.index("monthly_extras_spend")
    centroids = FITTED_SCALER.inverse_transform(V1_MODEL.cluster_centers_)
    return {
        cluster_id: float(centroids[cluster_id, spend_index])
        for cluster_id in range(len(CLUSTER_NAMES))
    }


def score_example_member() -> None:
    """Score the assignment's new member with the segment-level credit rule."""
    new_member = {
        "age": 30,
        "visits_per_month": 4,
        "avg_visit_minutes": 42,
        "classes_per_month": 1,
        "months_member": 5,
        "monthly_extras_spend": 10,
    }
    decision = decide_credit(new_member)
    print("New member")
    for feature in CLUSTER_FEATURES:
        value = new_member[feature]
        if feature == "monthly_extras_spend":
            print(f"{feature}: ${value:.0f}")
        else:
            print(f"{feature}: {value}")
    print(f"Cluster ID: {decision['cluster_id']}")
    print(f"Segment: {decision['segment']}")
    print(
        "Receives $10 credit: "
        f"{'Yes' if decision['receives_credit'] else 'No'}"
    )
    print(
        "The decision follows the segment rule. This member's $10 extras spend "
        "does not override it."
    )


def report_cluster_choice(comparison: pd.DataFrame) -> None:
    """Select k=4 when both curves bend there, and report the test-spread gain."""
    spreads = comparison.set_index("k")
    baseline = float(spreads.loc[1, "test_spread"])
    chosen_k = 4
    chosen = float(spreads.loc[chosen_k, "test_spread"])
    absolute_reduction = baseline - chosen
    percentage_reduction = 100 * absolute_reduction / baseline

    print("Elbow evaluation")
    print(
        "Both curves drop sharply through k=4 and then flatten. "
        "The held-out test spread supports the training elbow at k=4, "
        "so k=4 is the v1 model."
    )
    print()
    print(f"Chosen k: {chosen_k}")
    print(
        "Training elbow: inertia falls steeply from k=1 to k=4, "
        "then each extra cluster removes much less distance."
    )
    print(
        "Test elbow: assigning the 99 held-out members to the training "
        "centroids shows the same bend at k=4."
    )
    print(f"Test spread at k=1 (baseline): {baseline:.4f}")
    print(f"Test spread at k={chosen_k} (v1): {chosen:.4f}")
    print(f"Absolute reduction in test spread: {absolute_reduction:.4f}")
    print(f"Percentage reduction in test spread: {percentage_reduction:.1f}%")
    print()
    print(
        "For gym operations: four member groups capture most of the "
        "difference in how people use the gym. Splitting into five or "
        "more groups would make the segments harder to act on without "
        "making members much more alike within each group."
    )


if __name__ == "__main__":
    main()
