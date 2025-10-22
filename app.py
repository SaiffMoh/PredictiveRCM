from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import os
from pathlib import Path
import sys

# Load environment variables
load_dotenv()

# Import the claim processing classes
from claim_matcher import ClaimMatcher
from claim_justifier import ClaimJustifier

# Page configuration
st.set_page_config(
    page_title="Healthcare Claims Processing System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'matched_results' not in st.session_state:
    st.session_state.matched_results = None
if 'justified_results' not in st.session_state:
    st.session_state.justified_results = None

def get_available_providers():
    """Get list of available providers from the data/providers directory"""
    providers_dir = Path("data/providers")
    if not providers_dir.exists():
        return []
    return [d.name for d in providers_dir.iterdir() if d.is_dir()]

def get_provider_pricing_file(provider_name):
    """Get the pricing file path for a provider"""
    provider_dir = Path(f"data/providers/{provider_name}")
    if not provider_dir.exists():
        return None

    # Look for Excel files in the provider directory
    excel_files = list(provider_dir.glob("*.xlsx")) + list(provider_dir.glob("*.xls"))
    if excel_files:
        return str(excel_files[0])  # Return first Excel file found
    return None

def get_naphies_file():
    """Get the NAPHIES reference file"""
    naphies_dir = Path("data/naphies")
    if not naphies_dir.exists():
        return None

    excel_files = list(naphies_dir.glob("*.xlsx")) + list(naphies_dir.glob("*.xls"))
    if excel_files:
        return str(excel_files[0])
    return None

# Main UI
st.markdown('<h1 class="main-header">🏥 Healthcare Claims Processing System</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/1f77b4/ffffff?text=Claims+AI", width='stretch')
    st.markdown("---")
    st.markdown("### 📋 Processing Steps")
    st.markdown("""
    1. **Select Provider** - Choose pricing list
    2. **Upload Raw Data** - Upload claim data
    3. **Configure Settings** - Set parameters
    4. **Process Claims** - Run matching & justification
    5. **Download Results** - Export processed data
    """)
    st.markdown("---")
    st.markdown("### ℹ️ System Info")
    providers = get_available_providers()
    st.info(f"Available Providers: {len(providers)}")
    naphies_file = get_naphies_file()
    if naphies_file:
        st.success("✓ NAPHIES Reference Loaded")
    else:
        st.error("✗ NAPHIES Reference Missing")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🔧 Setup", "⚙️ Processing", "📊 Results", "📥 Download"])

with tab1:
    st.markdown('<div class="step-header">Step 1: Select Provider & Upload Data</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Provider Selection")
        providers = get_available_providers()

        if not providers:
            st.error("❌ No providers found in `data/providers/` directory")
            st.info("💡 Please create provider directories in `data/providers/` with their pricing lists")
        else:
            selected_provider = st.selectbox(
                "Select Provider:",
                options=providers,
                help="Choose the insurance provider for this batch of claims"
            )

            if selected_provider:
                pricing_file = get_provider_pricing_file(selected_provider)
                if pricing_file:
                    st.success(f"✓ Pricing list found: `{Path(pricing_file).name}`")
                    st.session_state.selected_provider = selected_provider
                    st.session_state.pricing_file = pricing_file
                else:
                    st.error(f"❌ No pricing file found for {selected_provider}")
                    st.info("💡 Add an Excel file to `data/providers/{selected_provider}/`")

    with col2:
        st.markdown("#### NAPHIES Reference")
        naphies_file = get_naphies_file()
        if naphies_file:
            st.success(f"✓ NAPHIES file: `{Path(naphies_file).name}`")
            st.session_state.naphies_file = naphies_file
        else:
            st.error("❌ NAPHIES reference file not found")
            st.info("💡 Add NAPHIES Excel file to `data/naphies/`")

    st.markdown("---")
    st.markdown("#### Upload Raw Claims Data")

    uploaded_file = st.file_uploader(
        "Upload Raw Claims Excel File",
        type=["xlsx", "xls"],
        help="Upload the raw claims data file for the selected provider"
    )

    if uploaded_file:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.raw_data_file = temp_path
        st.success(f"✓ Uploaded: {uploaded_file.name}")

        # Show preview
        try:
            preview_df = pd.read_excel(temp_path, sheet_name='Worksheet', nrows=5)
            # Normalize service_code to string
            preview_df['service_code'] = preview_df['service_code'].fillna('').astype(str)
            with st.expander("📄 Preview Raw Data (First 5 rows)"):
                st.dataframe(preview_df, width='stretch')
        except Exception as e:
            st.warning(f"Could not preview file: {e}")

with tab2:
    st.markdown('<div class="step-header">Step 2: Configure & Process Claims</div>', unsafe_allow_html=True)

    # Check if all required files are available
    ready_to_process = all([
        hasattr(st.session_state, 'selected_provider'),
        hasattr(st.session_state, 'pricing_file'),
        hasattr(st.session_state, 'naphies_file'),
        hasattr(st.session_state, 'raw_data_file')
    ])

    if not ready_to_process:
        st.warning("⚠️ Please complete Step 1 (Setup) before processing")
    else:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"""
        **Ready to Process:**
        - Provider: `{st.session_state.selected_provider}`
        - Raw Data: `{Path(st.session_state.raw_data_file).name}`
        - Pricing List: `{Path(st.session_state.pricing_file).name}`
        - NAPHIES Reference: `{Path(st.session_state.naphies_file).name}`
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### Processing Parameters")

        col1, col2 = st.columns(2)

        with col1:
            semantic_threshold = st.slider(
                "Semantic Matching Threshold",
                min_value=0.5,
                max_value=0.95,
                value=0.75,
                step=0.05,
                help="Minimum similarity score for Path 2 matching (higher = stricter)"
            )

            openai_api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Required for claim justification (GPT-4o-mini)",
                placeholder="sk-proj-...",
                value=os.getenv('OPENAI_API_KEY', '')
            )

        with col2:
            batch_size = st.number_input(
                "LLM Batch Size",
                min_value=5,
                max_value=50,
                value=10,
                help="Number of claims per API call"
            )

            max_workers = st.number_input(
                "Parallel Workers",
                min_value=1,
                max_value=10,
                value=5,
                help="Number of parallel threads for processing"
            )

        st.markdown("---")

        if st.button("🚀 Start Processing", type="primary", width='stretch'):
            if not openai_api_key:
                st.error("❌ OpenAI API Key is required for claim justification")
            else:
                # Create output directory
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)

                progress_placeholder = st.empty()
                status_placeholder = st.empty()

                try:
                    # Step 1: Claim Matching
                    with st.spinner("🔍 Running Claim Matcher..."):
                        status_placeholder.info("**Phase 1/2:** Matching claims with NAPHIES codes...")

                        matcher = ClaimMatcher()
                        matched_results = matcher.run_pipeline(
                            raw_file=st.session_state.raw_data_file,
                            pricing_file=st.session_state.pricing_file,
                            naphies_file=st.session_state.naphies_file,
                            output_file='output/matched_claims.xlsx',
                            no_match_file='output/no_match_claims.xlsx',
                            review_file='output/needs_review_claims.xlsx',
                            semantic_threshold=semantic_threshold
                        )

                        st.session_state.matched_results = matched_results
                        status_placeholder.success("✅ Phase 1 Complete: Claims matched successfully!")

                    # Step 2: Claim Justification
                    with st.spinner("⚖️ Running Claim Justifier..."):
                        status_placeholder.info("**Phase 2/2:** Justifying claims with AI...")

                        justifier = ClaimJustifier(
                            api_key=openai_api_key,
                            batch_size=batch_size,
                            max_workers=max_workers
                        )

                        justified_results = justifier.run_justification_pipeline(
                            input_file='output/matched_claims.xlsx',
                            output_file='output/justified_claims.xlsx',
                            filter_path=None
                        )

                        st.session_state.justified_results = justified_results
                        st.session_state.processing_complete = True
                        status_placeholder.success("✅ Phase 2 Complete: Claims justified successfully!")

                    st.balloons()
                    st.success("🎉 **Processing Complete!** Check the Results tab for details.")

                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)
                finally:
                    # Cleanup temp file
                    if os.path.exists(st.session_state.raw_data_file):
                        os.remove(st.session_state.raw_data_file)

with tab3:
    st.markdown('<div class="step-header">Step 3: View Results</div>', unsafe_allow_html=True)

    if not st.session_state.processing_complete:
        st.info("⏳ No results yet. Please complete processing in the Processing tab.")
    else:
        results = st.session_state.justified_results

        # Summary Statistics
        st.markdown("#### 📊 Processing Summary")

        col1, col2, col3, col4 = st.columns(4)

        total_claims = len(results)
        path1_count = (results['match_path'] == 'Path1_Exact').sum()
        path2_count = (results['match_path'] == 'Path2_Matched').sum()
        no_match_count = (results['match_path'] == 'No_Match').sum()

        justified_count = (results['status'] == 'Justified').sum()
        not_justified_count = (results['status'] == 'Not Justified').sum()
        needs_review_count = results['needs_manual_review'].sum()

        with col1:
            st.metric("Total Claims", f"{total_claims:,}")
        with col2:
            st.metric("Matched", f"{path1_count + path2_count:,}",
                     f"{((path1_count + path2_count)/total_claims*100):.1f}%")
        with col3:
            st.metric("Justified", f"{justified_count:,}",
                     f"{(justified_count/total_claims*100):.1f}%")
        with col4:
            st.metric("Needs Review", f"{needs_review_count:,}",
                     f"{(needs_review_count/total_claims*100):.1f}%")

        st.markdown("---")

        # Detailed Breakdown
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Match Path Distribution")
            match_dist = results['match_path'].value_counts()
            st.dataframe(
                pd.DataFrame({
                    'Match Path': match_dist.index,
                    'Count': match_dist.values,
                    'Percentage': [f"{v/total_claims*100:.1f}%" for v in match_dist.values]
                }),
                width='stretch',
                hide_index=True
            )

        with col2:
            st.markdown("##### Justification Status")
            status_dist = results['status'].value_counts()
            st.dataframe(
                pd.DataFrame({
                    'Status': status_dist.index,
                    'Count': status_dist.values,
                    'Percentage': [f"{v/total_claims*100:.1f}%" for v in status_dist.values]
                }),
                width='stretch',
                hide_index=True
            )

        st.markdown("---")

        # Interactive Data Table
        st.markdown("#### 🔍 Detailed Results")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            filter_match = st.multiselect(
                "Filter by Match Path:",
                options=results['match_path'].unique(),
                default=results['match_path'].unique()
            )

        with col2:
            filter_status = st.multiselect(
                "Filter by Status:",
                options=results['status'].unique(),
                default=results['status'].unique()
            )

        with col3:
            filter_review = st.selectbox(
                "Filter by Review Flag:",
                options=['All', 'Needs Review', 'No Review Needed']
            )

        # Apply filters
        filtered_results = results[
            (results['match_path'].isin(filter_match)) &
            (results['status'].isin(filter_status))
        ]

        if filter_review == 'Needs Review':
            filtered_results = filtered_results[filtered_results['needs_manual_review'] == True]
        elif filter_review == 'No Review Needed':
            filtered_results = filtered_results[filtered_results['needs_manual_review'] == False]

        st.dataframe(
            filtered_results[[
                'needs_manual_review', 'status', 'match_path', 'match_score',
                'naphies_code', 'service_description', 'Diagnoses', 'note'
            ]].head(100),
            width='stretch'
        )

        st.caption(f"Showing {len(filtered_results):,} of {total_claims:,} claims (limited to 100 rows)")

with tab4:
    st.markdown('<div class="step-header">Step 4: Download Results</div>', unsafe_allow_html=True)

    if not st.session_state.processing_complete:
        st.info("⏳ No results available for download. Please complete processing first.")
    else:
        st.markdown("#### 📥 Download Processed Files")

        col1, col2, col3 = st.columns(3)

        with col1:
            if os.path.exists('output/justified_claims.xlsx'):
                with open('output/justified_claims.xlsx', 'rb') as f:
                    st.download_button(
                        label="📄 Download All Results",
                        data=f,
                        file_name=f"{st.session_state.selected_provider}_justified_claims.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width='stretch'
                    )

        with col2:
            if os.path.exists('output/no_match_claims.xlsx'):
                with open('output/no_match_claims.xlsx', 'rb') as f:
                    st.download_button(
                        label="❌ Download No-Match Claims",
                        data=f,
                        file_name=f"{st.session_state.selected_provider}_no_match.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width='stretch'
                    )

        with col3:
            # Create a filtered file with only claims needing review
            if st.session_state.justified_results is not None:
                review_claims = st.session_state.justified_results[
                    st.session_state.justified_results['needs_manual_review'] == True
                ]
                review_path = 'output/manual_review_required.xlsx'
                review_claims.to_excel(review_path, index=False)

                with open(review_path, 'rb') as f:
                    st.download_button(
                        label="⚠️ Download Review Required",
                        data=f,
                        file_name=f"{st.session_state.selected_provider}_needs_review.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width='stretch'
                    )

        st.markdown("---")

        st.markdown("#### 📋 File Descriptions")
        st.markdown("""
        - **All Results**: Complete dataset with matching, enrichment, and justification
        - **No-Match Claims**: Claims that couldn't be matched to NAPHIES codes
        - **Review Required**: Claims flagged for manual review (Path 2 matches, edge cases, IV infusions)
        """)

        if st.button("🔄 Reset & Start New Batch", type="secondary", width='stretch'):
            st.session_state.processing_complete = False
            st.session_state.matched_results = None
            st.session_state.justified_results = None
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Healthcare Claims Processing System v1.0 | Powered by AI 🤖"
    "</div>",
    unsafe_allow_html=True
)
