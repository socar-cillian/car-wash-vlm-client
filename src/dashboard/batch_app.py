"""Streamlit dashboard for visualizing batch inference results."""

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


def load_batch_results(csv_path: Path) -> pd.DataFrame:
    """Load batch inference results from CSV."""
    df = pd.read_csv(csv_path)
    return df


def get_image_path(image_name: str, images_dir: Path) -> Path:
    """Find image path recursively in the images directory."""
    # Try direct path first
    direct_path = images_dir / image_name
    if direct_path.exists():
        return direct_path

    # Search recursively
    for img_path in images_dir.rglob(image_name):
        return img_path

    return None


def parse_contamination_list(value) -> str:
    """Parse contamination types or parts list for display."""
    if pd.isna(value) or value == "":
        return "없음"

    # If it's already a string, return it
    if isinstance(value, str):
        # Remove brackets and quotes if present
        value = value.strip("[]'\"")
        if value == "":
            return "없음"
        return value

    return str(value)


def main():
    st.set_page_config(page_title="Batch Inference Dashboard", page_icon="🚗", layout="wide")

    # Custom CSS
    st.markdown(
        """
        <style>
        /* Limit image height */
        [data-testid="stImage"] img {
            max-height: 600px;
            object-fit: contain;
        }

        /* Card styling for results */
        .result-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }

        /* Classification badges */
        .badge-normal {
            background-color: #28a745;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }

        .badge-dirty {
            background-color: #dc3545;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🚗 Batch Inference Results Dashboard")
    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ 설정")

        # File paths
        csv_path = st.text_input(
            "결과 CSV 경로", value="results/output.csv", help="Batch inference results CSV file path"
        )
        images_dir = st.text_input(
            "이미지 디렉토리", value="images/sample_images/images", help="Directory containing images"
        )

        # Load data button
        if st.button("🔄 데이터 로드", type="primary"):
            st.session_state.reload = True

    # Load data
    csv_file = Path(csv_path)
    img_dir = Path(images_dir)

    if not csv_file.exists():
        st.error(f"❌ CSV 파일을 찾을 수 없습니다: {csv_file}")
        st.info("💡 올바른 CSV 파일 경로를 입력해주세요.")
        return

    if not img_dir.exists():
        st.error(f"❌ 이미지 디렉토리를 찾을 수 없습니다: {img_dir}")
        st.info("💡 올바른 이미지 디렉토리 경로를 입력해주세요.")
        return

    # Load results
    try:
        df = load_batch_results(csv_file)
    except Exception as e:
        st.error(f"❌ CSV 파일을 로드하는 중 오류가 발생했습니다: {e}")
        return

    # Summary statistics
    st.header("📊 요약 통계")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("전체 이미지", len(df))
    with col2:
        success_count = df["success"].sum() if "success" in df.columns else 0
        st.metric("성공", success_count)
    with col3:
        normal_count = (df["classification"] == "Normal").sum() if "classification" in df.columns else 0
        st.metric("🟢 Normal", normal_count)
    with col4:
        dirty_count = (df["classification"] == "Dirty").sum() if "classification" in df.columns else 0
        st.metric("🔴 Dirty", dirty_count)

    st.markdown("---")

    # Filters
    st.header("🔍 필터")
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        # Classification filter
        classification_options = ["전체"] + sorted(df["classification"].dropna().unique().tolist())
        classification_filter = st.multiselect(
            "Classification",
            options=classification_options,
            default=["전체"],
            help="필터링할 classification을 선택하세요",
        )

    with filter_col2:
        # Success filter
        success_filter = st.selectbox("추론 성공 여부", options=["전체", "성공", "실패"], index=0)

    with filter_col3:
        # Model filter
        if "model" in df.columns:
            model_options = ["전체"] + sorted(df["model"].dropna().unique().tolist())
            model_filter = st.selectbox("모델", options=model_options, index=0)
        else:
            model_filter = "전체"

    # Apply filters
    filtered_df = df.copy()

    # Classification filter
    if "classification" in df.columns and "전체" not in classification_filter:
        filtered_df = filtered_df[filtered_df["classification"].isin(classification_filter)]

    # Success filter
    if success_filter == "성공":
        filtered_df = filtered_df[filtered_df["success"]]
    elif success_filter == "실패":
        filtered_df = filtered_df[~filtered_df["success"]]

    # Model filter
    if model_filter != "전체" and "model" in df.columns:
        filtered_df = filtered_df[filtered_df["model"] == model_filter]

    st.info(f"📊 필터링된 결과: {len(filtered_df)} / {len(df)} 이미지")

    st.markdown("---")

    # Image navigation
    st.header("🖼️ 이미지 결과")

    if len(filtered_df) == 0:
        st.warning("⚠️ 표시할 결과가 없습니다. 필터를 조정해주세요.")
        return

    # Image selector
    col_selector1, col_selector2 = st.columns([3, 1])

    with col_selector1:
        image_idx = st.slider(
            "이미지 선택",
            min_value=0,
            max_value=len(filtered_df) - 1,
            value=0,
            help="슬라이더를 움직여 다른 이미지를 확인하세요",
        )

    with col_selector2:
        st.metric("현재 이미지", f"{image_idx + 1} / {len(filtered_df)}")

    row = filtered_df.iloc[image_idx]
    image_name = row["image_name"]
    image_path = get_image_path(image_name, img_dir)

    # Display image and results
    col_img, col_results = st.columns([1, 1])

    with col_img:
        st.subheader(f"📷 이미지: {image_name}")
        if image_path and image_path.exists():
            try:
                image = Image.open(image_path)
                st.image(image, use_container_width=True, output_format="auto")

                # Add expander for full-size image view
                with st.expander("🔍 이미지 확대 보기"):
                    st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f"❌ 이미지를 로드하는 중 오류가 발생했습니다: {e}")
        else:
            st.error(f"❌ 이미지를 찾을 수 없습��다: {image_name}")
            st.info(f"💡 검색 경로: {img_dir}")

    with col_results:
        st.subheader("📋 분석 결과")

        # Classification badge
        classification = row.get("classification", "N/A")
        if classification == "Normal":
            st.markdown('<span class="badge-normal">🟢 Normal</span>', unsafe_allow_html=True)
        elif classification == "Dirty":
            st.markdown('<span class="badge-dirty">🔴 Dirty</span>', unsafe_allow_html=True)
        else:
            st.markdown(f"**Classification**: {classification}")

        st.markdown("---")

        # Basic info
        st.markdown("### 📝 기본 정보")
        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.markdown(f"**모델**: `{row.get('model', 'N/A')}`")
            st.markdown(f"**추론 성공**: {'✅' if row.get('success', False) else '❌'}")

        with info_col2:
            latency = row.get("latency_seconds", 0)
            st.markdown(f"**처리 시간**: `{latency:.3f}초`")
            if row.get("error"):
                st.markdown(f"**에러**: {row.get('error', 'N/A')}")

        st.markdown("---")

        # Contamination details
        if classification == "Dirty":
            st.markdown("### 🔴 오염 상세 정보")

            contamination_types = parse_contamination_list(row.get("contamination_types", ""))
            contamination_parts = parse_contamination_list(row.get("contamination_parts", ""))

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown("**오염 유형**")
                st.info(contamination_types)

            with detail_col2:
                st.markdown("**오염 부위**")
                st.info(contamination_parts)
        else:
            st.markdown("### 🟢 오염 없음")
            st.success("이 이미지는 오염이 감지되지 않았습니다.")

        st.markdown("---")

        # Raw response
        with st.expander("🔧 Raw Response 보기"):
            raw_response = row.get("raw_response", "N/A")
            st.code(raw_response, language="python")

    st.markdown("---")

    # Statistics by classification
    st.header("📈 Classification 통계")

    stat_col1, stat_col2 = st.columns(2)

    with stat_col1:
        st.markdown("### Classification 분포")
        if "classification" in df.columns:
            classification_counts = df["classification"].value_counts()
            st.bar_chart(classification_counts)
        else:
            st.info("Classification 데이터가 없습니다.")

    with stat_col2:
        st.markdown("### 평균 처리 시간")
        if "latency_seconds" in df.columns and "classification" in df.columns:
            avg_latency = df.groupby("classification")["latency_seconds"].mean()
            st.bar_chart(avg_latency)
        else:
            st.info("처리 시간 데이터가 없습니다.")

    st.markdown("---")

    # Raw data view
    with st.expander("📋 전체 데이터 보기"):
        st.dataframe(filtered_df, use_container_width=True)

    # Download filtered results
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 필터링된 결과 다운로드 (CSV)",
        data=csv_data,
        file_name=f"filtered_batch_results_{len(filtered_df)}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
