"""Streamlit dashboard for visualizing car contamination inference results."""

import csv
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


def load_inference_results(csv_path: Path) -> pd.DataFrame:
    """Load inference results from CSV."""
    df = pd.read_csv(csv_path)
    return df


def display_area_results(row: pd.Series, areas: list[str]) -> None:
    """Display contamination results for each area."""
    cols = st.columns(len(areas))

    for idx, area in enumerate(areas):
        with cols[idx]:
            contamination_type = row.get(f"{area}_contamination_type", "N/A")
            severity = row.get(f"{area}_severity", "N/A")

            # Color code based on severity
            if severity == "심각":
                color = "🔴"
            elif severity == "보통":
                color = "🟡"
            elif severity == "양호":
                color = "🟢"
            else:
                color = "⚪"

            st.markdown(f"**{area}** {color}")
            st.caption(f"오염: {contamination_type}")
            st.caption(f"정도: {severity}")


def main():
    st.set_page_config(
        page_title="Car Contamination Dashboard",
        page_icon="🚗",
        layout="wide"
    )

    # Custom CSS for image height control
    st.markdown("""
        <style>
        /* Limit image height to match results column */
        [data-testid="stImage"] img {
            max-height: 600px;
            object-fit: contain;
        }

        /* Make expander images full width */
        [data-testid="stExpander"] [data-testid="stImage"] img {
            max-height: none;
            object-fit: contain;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🚗 Car Contamination Classification Dashboard")
    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.header("설정")

        # File paths
        csv_path = st.text_input(
            "결과 CSV 경로",
            value="results/inference_results.csv",
            help="Inference results CSV file path"
        )
        images_dir = st.text_input(
            "이미지 디렉토리",
            value="images/sample_images/images",
            help="Directory containing images"
        )

        # Load data button
        if st.button("데이터 로드", type="primary"):
            st.session_state.reload = True

    # Load data
    csv_file = Path(csv_path)
    img_dir = Path(images_dir)

    if not csv_file.exists():
        st.error(f"CSV 파일을 찾을 수 없습니다: {csv_file}")
        return

    if not img_dir.exists():
        st.error(f"이미지 디렉토리를 찾을 수 없습니다: {img_dir}")
        return

    # Load results
    df = load_inference_results(csv_file)

    # Summary statistics
    st.header("📊 요약 통계")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("전체 이미지", len(df))
    with col2:
        success_count = df["success"].sum() if "success" in df.columns else 0
        st.metric("성공", success_count)
    with col3:
        if "image_type" in df.columns:
            interior_count = (df["image_type"] == "내부").sum()
            st.metric("내부", interior_count)
        else:
            st.metric("내부", "N/A")
    with col4:
        if "image_type" in df.columns:
            exterior_count = (df["image_type"] == "외부").sum()
            st.metric("외부", exterior_count)
        else:
            st.metric("외부", "N/A")

    st.markdown("---")

    # Filters
    st.header("🔍 필터")
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        image_type_filter = st.multiselect(
            "이미지 타입",
            options=df["image_type"].unique() if "image_type" in df.columns else [],
            default=df["image_type"].unique() if "image_type" in df.columns else []
        )

    with filter_col2:
        gt_area_filter = st.multiselect(
            "GT 영역",
            options=df["gt_contamination_area"].unique() if "gt_contamination_area" in df.columns else [],
            default=df["gt_contamination_area"].unique() if "gt_contamination_area" in df.columns else []
        )

    with filter_col3:
        success_filter = st.selectbox(
            "추론 성공 여부",
            options=["전체", "성공", "실패"],
            index=0
        )

    # Apply filters
    filtered_df = df.copy()
    if image_type_filter and "image_type" in df.columns:
        filtered_df = filtered_df[filtered_df["image_type"].isin(image_type_filter)]
    if gt_area_filter and "gt_contamination_area" in df.columns:
        filtered_df = filtered_df[filtered_df["gt_contamination_area"].isin(gt_area_filter)]
    if success_filter == "성공":
        filtered_df = filtered_df[filtered_df["success"] == True]
    elif success_filter == "실패":
        filtered_df = filtered_df[filtered_df["success"] == False]

    st.info(f"필터링된 결과: {len(filtered_df)} / {len(df)} 이미지")

    st.markdown("---")

    # Image navigation
    st.header("🖼️ 이미지 결과")

    if len(filtered_df) == 0:
        st.warning("표시할 결과가 없습니다.")
        return

    # Image selector
    image_idx = st.slider(
        "이미지 선택",
        min_value=0,
        max_value=len(filtered_df) - 1,
        value=0
    )

    row = filtered_df.iloc[image_idx]
    image_name = row["image_name"]
    image_path = img_dir / image_name

    # Display image and results
    col_img, col_results = st.columns([1, 1])

    with col_img:
        st.subheader(f"이미지: {image_name}")
        if image_path.exists():
            image = Image.open(image_path)

            # Display image with fixed height to match results column
            # Use a container with max-height and enable click to expand
            st.image(image, use_container_width=True, output_format="auto")

            # Add expander for full-size image view
            with st.expander("🔍 이미지 확대 보기"):
                st.image(image, use_container_width=True)
        else:
            st.error(f"이미지를 찾을 수 없습니다: {image_path}")

    with col_results:
        st.subheader("분석 결과")

        # Basic info
        st.markdown("### 기본 정보")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown(f"**모델**: {row.get('model', 'N/A')}")
            st.markdown(f"**처리 시간**: {row.get('latency_seconds', 'N/A'):.3f}초")
        with info_col2:
            st.markdown(f"**추론 성공**: {'✅' if row.get('success', False) else '❌'}")
            st.markdown(f"**이미지 타입**: {row.get('image_type', 'N/A')}")

        # Ground truth
        st.markdown("### Ground Truth")
        gt_col1, gt_col2 = st.columns(2)
        with gt_col1:
            st.markdown(f"**영역**: {row.get('gt_contamination_area', 'N/A')}")
        with gt_col2:
            st.markdown(f"**오염 타입**: {row.get('gt_contamination_type', 'N/A')}")

        st.markdown("---")

        # Area-specific results
        st.markdown("### 영역별 상세 결과")

        # Determine which areas to show based on image type
        image_type = row.get("image_type", "")

        if image_type == "내부":
            areas = ["운전석", "조수석", "컵홀더", "뒷좌석"]
            st.markdown("**내부 영역**")
            display_area_results(row, areas)
        elif image_type == "외부":
            areas = ["전면", "조수석_방향", "운전석_방향", "후면"]
            st.markdown("**외부 영역**")
            display_area_results(row, areas)
        else:
            st.info("이미지 타입이 지정되지 않았거나 관련없음으로 분류되었습니다.")

    st.markdown("---")

    # Raw data view
    with st.expander("📋 전체 데이터 보기"):
        st.dataframe(filtered_df, use_container_width=True)

    # Download filtered results
    st.download_button(
        label="필터링된 결과 다운로드 (CSV)",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name=f"filtered_results_{len(filtered_df)}.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
