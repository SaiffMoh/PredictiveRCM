from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import io
import uuid
import tempfile
import atexit
import re
import json
# Load environment variables
load_dotenv()

# Import the claim processing classes
from claim_matcher import ClaimMatcher
from claim_justifier import ClaimJustifier
from drug_matcher import DrugMatcher
from drug_justifier import DrugJustifier

def save_to_temp_file(df, temp_path):
    with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)  # Explicitly close the writer to release file handle

def try_delete_file(file_path, retries=5, delay=2):
    for attempt in range(retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except PermissionError:
            if attempt == retries - 1:
                return False
            time.sleep(delay)
    return False


import streamlit as st
import sys
from io import StringIO
import contextlib

@contextlib.contextmanager
def capture_output():
    """Capture stdout to display in Streamlit"""
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    try:
        yield captured_output
    finally:
        sys.stdout = old_stdout

def process_services_with_progress(raw_file, pricing_file, naphies_file, semantic_threshold, 
                                   openai_api_key, batch_size, max_workers, status_placeholder, sheet_name=None):
    """Process service claims with progress updates"""
    try:
        status_placeholder.info("🔄 Step 1/2: Running ClaimMatcher (Path 1 & Path 2)...")
        
        # Capture matcher output
        with capture_output() as output:
            matcher = ClaimMatcher()
            # Pass sheet_name to the matcher
            matched_results = matcher.run_pipeline(
                raw_file=raw_file,
                pricing_file=pricing_file,
                naphies_file=naphies_file,
                output_file='output/service_matched_claims.xlsx',
                no_match_file='output/service_no_match_claims.xlsx',
                review_file='output/service_needs_review_claims.xlsx',
                semantic_threshold=semantic_threshold,
                sheet_name=sheet_name
            )
        
        # Show matcher output
        matcher_output = output.getvalue()
        if matcher_output:
            with st.expander("📋 ClaimMatcher Output", expanded=False):
                st.code(matcher_output, language='text')
        
        status_placeholder.success("✅ Step 1/2: Service matching complete!")
        status_placeholder.info("🔄 Step 2/2: Running ClaimJustifier (AI Analysis)...")
        
        # Capture justifier output
        with capture_output() as output:
            justifier = ClaimJustifier(
                api_key=openai_api_key,
                batch_size=batch_size,
                max_workers=max_workers
            )
            justified_results = justifier.run_justification_pipeline(
                input_file='output/service_matched_claims.xlsx',
                output_file='output/service_justified_claims.xlsx',
                filter_path=None
            )
        
        # Show justifier output
        justifier_output = output.getvalue()
        if justifier_output:
            with st.expander("🤖 ClaimJustifier Output", expanded=False):
                st.code(justifier_output, language='text')
        
        status_placeholder.success("✅ Step 2/2: Service justification complete!")
        
        return {'matched': matched_results, 'justified': justified_results, 'status': 'success'}
    
    except Exception as e:
        status_placeholder.error(f"❌ Service processing failed: {str(e)}")
        return {'status': 'error', 'error': str(e)}


def process_drugs_with_progress(raw_file, sfda_file, chi_file, openai_api_key, 
                                batch_size, max_workers, status_placeholder):
    """Process drug claims with progress updates"""
    try:
        status_placeholder.info("🔄 Step 1/2: Running DrugMatcher (SFDA + CHI)...")
        
        # Capture matcher output
        with capture_output() as output:
            matcher = DrugMatcher()
            matched_results = matcher.run_pipeline(
                raw_file=raw_file,
                sfda_file=sfda_file,
                chi_file=chi_file,
                output_file='output/drug_matched_claims.xlsx',
                no_match_file='output/drug_no_match_claims.xlsx',
                review_file='output/drug_needs_review_claims.xlsx'
            )
        
        # Show matcher output
        matcher_output = output.getvalue()
        if matcher_output:
            with st.expander("📋 DrugMatcher Output", expanded=False):
                st.code(matcher_output, language='text')
        
        status_placeholder.success("✅ Step 1/2: Drug matching complete!")
        status_placeholder.info("🔄 Step 2/2: Running DrugJustifier (AI Analysis)...")
        
        # Capture justifier output
        with capture_output() as output:
            justifier = DrugJustifier(
                api_key=openai_api_key,
                batch_size=batch_size,
                max_workers=max_workers
            )
            justified_results = justifier.run_justification_pipeline(
                input_file='output/drug_matched_claims.xlsx',
                output_file='output/drug_justified_claims.xlsx',
                filter_path=None
            )
        
        # Show justifier output
        justifier_output = output.getvalue()
        if justifier_output:
            with st.expander("🤖 DrugJustifier Output", expanded=False):
                st.code(justifier_output, language='text')
        
        status_placeholder.success("✅ Step 2/2: Drug justification complete!")
        
        return {'matched': matched_results, 'justified': justified_results, 'status': 'success'}
    
    except Exception as e:
        status_placeholder.error(f"❌ Drug processing failed: {str(e)}")
        return {'status': 'error', 'error': str(e)}
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
    .drug-badge {
        background-color: #ff6b6b;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
    }
    .service-badge {
        background-color: #4ecdc4;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'service_matched_results' not in st.session_state:
    st.session_state.service_matched_results = None
if 'service_justified_results' not in st.session_state:
    st.session_state.service_justified_results = None
if 'drug_matched_results' not in st.session_state:
    st.session_state.drug_matched_results = None
if 'drug_justified_results' not in st.session_state:
    st.session_state.drug_justified_results = None
if 'claim_types' not in st.session_state:
    st.session_state.claim_types = None
if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = 'auto'
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = None
if 'filtered_service_data' not in st.session_state:
    st.session_state.filtered_service_data = None
if 'filtered_drug_data' not in st.session_state:
    st.session_state.filtered_drug_data = None

# Required columns for services and drugs
REQUIRED_SERVICE_COLUMNS = [
    'company', 'service_code', 'service_description', 'gender',
    'ICD10 Code', 'Diagnoses', 'chief_complaint'
]

REQUIRED_DRUG_COLUMNS = [
    'company', 'service_code', 'service_description', 'gender',
    'ICD10 Code', 'Diagnoses', 'chief_complaint'
]

# Common column name variations for preprocessing
COLUMN_VARIATIONS = {
    'company': ['insurance', 'insurer', 'provider', 'company_name'],
    'service_code': ['code', 'servicecode', 'service_code', 'payer_code', 'his_code'],
    'service_description': ['description', 'servicedesc', 'service_desc', 'his_description'],
    'gender': ['sex', 'patient_gender'],
    'ICD10 Code': ['icd10', 'icd10code', 'diagnosis_code', 'icd_code'],
    'Diagnoses': ['diagnosis', 'patient_diagnosis', 'diagnoses'],
    'chief_complaint': ['complaint', 'chiefcomplaint', 'patient_complaint']
}

def cleanup_temp_files():
    """Clean up temporary files on app exit."""
    for temp_file in ['service_temp_file', 'drug_temp_file', 'raw_data_file']:
        if temp_file in st.session_state and os.path.exists(st.session_state[temp_file]):
            if not try_delete_file(st.session_state[temp_file]):
                st.warning(f"Could not delete {st.session_state[temp_file]}. It may be in use.")  # Or use st.warning if in Streamlit context

atexit.register(cleanup_temp_files)

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
    excel_files = list(provider_dir.glob("*.xlsx")) + list(provider_dir.glob("*.xls"))
    if excel_files:
        return str(excel_files[0])
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

def get_sfda_file():
    """Get the SFDA reference file"""
    sfda_dir = Path("data/sfda")
    if not sfda_dir.exists():
        return None
    excel_files = list(sfda_dir.glob("*.xlsx")) + list(sfda_dir.glob("*.xls"))
    if excel_files:
        return str(excel_files[0])
    return None

def get_chi_file():
    """Get the CHI reference file"""
    chi_dir = Path("data/chi")
    if not chi_dir.exists():
        return None
    excel_files = list(chi_dir.glob("*.xlsx")) + list(chi_dir.glob("*.xls"))
    if excel_files:
        return str(excel_files[0])
    return None

def detect_claim_types(df):
    """
    Detect if uploaded data contains drugs, services, or both
    Returns dict with counts
    """
    if 'service_type' not in df.columns:
        return {'drugs': 0, 'services': len(df), 'total': len(df)}

    medicine_keywords = ['medicine', 'drug', 'pharmacy', 'medication']
    drug_mask = df['service_type'].str.lower().str.contains(
        '|'.join(medicine_keywords),
        na=False
    )

    drugs_count = drug_mask.sum()
    services_count = (~drug_mask).sum()

    return {
        'drugs': drugs_count,
        'services': services_count,
        'total': len(df)
    }

def preprocess_column_mapping(df):
    """
    Attempt to automatically map user columns to required columns
    using common naming conventions
    """
    mapping = {}
    user_columns = df.columns.tolist()

    for required_col, variations in COLUMN_VARIATIONS.items():
        for variation in variations:
            if variation in user_columns:
                mapping[required_col] = variation
                break
        else:
            # If no variation found, try fuzzy matching
            for col in user_columns:
                if required_col.lower() in col.lower() or col.lower() in required_col.lower():
                    mapping[required_col] = col
                    break
            else:
                mapping[required_col] = None

    return mapping

def process_services(raw_file, pricing_file, naphies_file, semantic_threshold, 
                    openai_api_key, batch_size, max_workers):
    """Wrapper for backward compatibility"""
    return process_services_with_progress(
        raw_file, pricing_file, naphies_file, semantic_threshold,
        openai_api_key, batch_size, max_workers, 
        st.empty()  # Dummy placeholder
    )

def process_drugs(raw_file, sfda_file, chi_file, openai_api_key, batch_size, max_workers):
    """Wrapper for backward compatibility"""
    return process_drugs_with_progress(
        raw_file, sfda_file, chi_file, openai_api_key,
        batch_size, max_workers,
        st.empty()  # Dummy placeholder
    )

# Main UI
st.markdown('<h1 class="main-header">🏥 Healthcare Claims Processing System</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/1f77b4/ffffff?text=Claims+AI", use_container_width=True)
    st.markdown("---")
    st.markdown("### 📋 Processing Steps")
    st.markdown("""
    1. **Select Provider** - Choose pricing list
    2. **Upload Raw Data** - Upload claim data
    3. **Map Columns** - Match your columns to required columns
    4. **Configure Settings** - Set parameters
    5. **Process Claims** - Run matching & justification
    6. **Download Results** - Export processed data
    """)
    st.markdown("---")
    st.markdown("### ℹ️ System Info")
    providers = get_available_providers()
    st.info(f"Available Providers: {len(providers)}")

    # Reference files status
    naphies_file = get_naphies_file()
    sfda_file = get_sfda_file()
    chi_file = get_chi_file()

    if naphies_file:
        st.success("✓ NAPHIES (Services)")
    else:
        st.error("✗ NAPHIES Missing")

    if sfda_file:
        st.success("✓ SFDA (Drugs)")
    else:
        st.error("✗ SFDA Missing")

    if chi_file:
        st.success("✓ CHI (Drugs)")
    else:
        st.error("✗ CHI Missing")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔧 Setup", "🗂️ Column Mapper", "⚙️ Processing", "📊 Results", "📥 Download"])

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
                    st.info(f"💡 Add an Excel file to `data/providers/{selected_provider}/`")

    with col2:
        st.markdown("#### Reference Files")

        # NAPHIES
        naphies_file = get_naphies_file()
        if naphies_file:
            st.success(f"✓ NAPHIES: `{Path(naphies_file).name}`")
            st.session_state.naphies_file = naphies_file
        else:
            st.error("❌ NAPHIES reference file not found")

        # SFDA
        sfda_file = get_sfda_file()
        if sfda_file:
            st.success(f"✓ SFDA: `{Path(sfda_file).name}`")
            st.session_state.sfda_file = sfda_file
        else:
            st.warning("⚠️ SFDA reference file not found (for drugs)")

        # CHI
        chi_file = get_chi_file()
        if chi_file:
            st.success(f"✓ CHI: `{Path(chi_file).name}`")
            st.session_state.chi_file = chi_file
        else:
            st.warning("⚠️ CHI reference file not found (for drugs)")

    st.markdown("---")
    st.markdown("#### Upload Raw Claims Data")
    uploaded_file = st.file_uploader(
        "Upload Raw Claims Excel File",
        type=["xlsx", "xls"],
        help="Upload the raw claims data file (services and/or drugs)"
    )

    if uploaded_file:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(uploaded_file.getbuffer())
        st.session_state.raw_data_file = tmp_path

        # Read Excel file to get sheet names
        try:
            excel_file = pd.ExcelFile(tmp_path)
            sheet_names = excel_file.sheet_names
            if len(sheet_names) > 1:
                st.session_state.selected_sheet = st.selectbox(
                    "Select the sheet to work on:",
                    options=sheet_names,
                    help="Choose the sheet containing your claims data"
                )
            else:
                st.session_state.selected_sheet = sheet_names[0]
                st.info(f"Only one sheet found: '{st.session_state.selected_sheet}'")

            # Load the selected sheet
            df = pd.read_excel(tmp_path, sheet_name=st.session_state.selected_sheet)
            st.session_state.raw_data = df
            st.success(f"✓ Uploaded: {uploaded_file.name} (Sheet: {st.session_state.selected_sheet})")

            # Detect claim types and show summary
            try:
                preview_df = df
                claim_types = detect_claim_types(preview_df)
                st.session_state.claim_types = claim_types

                # Display claim type breakdown
                st.markdown("#### Detected Claim Types")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Total Claims", f"{claim_types['total']:,}")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<span class="service-badge">SERVICES</span>', unsafe_allow_html=True)
                    st.metric("Service Claims", f"{claim_types['services']:,}")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<span class="drug-badge">DRUGS</span>', unsafe_allow_html=True)
                    st.metric("Drug Claims", f"{claim_types['drugs']:,}")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Show preview
                preview_df['service_code'] = preview_df['service_code'].fillna('').astype(str)
                with st.expander("📄 Preview Raw Data (First 5 rows)"):
                    st.dataframe(preview_df.head(5), use_container_width=True)

            except Exception as e:
                st.warning(f"Could not preview file: {e}")

        except Exception as e:
            st.error(f"Error reading Excel file: {e}")

with tab2:
    st.markdown('<div class="step-header">Step 2: Map Columns</div>', unsafe_allow_html=True)

    if 'raw_data' not in st.session_state:
        st.warning("⚠️ Please upload a file in the Setup tab first.")
    else:
        df = st.session_state.raw_data
        user_columns = df.columns.tolist()

        st.markdown("#### Required Columns")
        st.markdown("""
        The system requires the following columns for processing:
        - **Services**: `company`, `service_code`, `service_description`, `gender`, `ICD10 Code`, `Diagnoses`, `chief_complaint`
        - **Drugs**: `company`, `service_code`, `service_description`, `gender`, `ICD10 Code`, `Diagnoses`, `chief_complaint`
        """)

        st.markdown("#### Your Columns")
        st.write(f"Your file has the following columns: **{', '.join(user_columns)}**")

        # Preprocess column mapping
        st.markdown("#### Auto-Mapping")
        auto_mapping = preprocess_column_mapping(df)
        st.write("Suggested column mapping:")
        mapping_df = pd.DataFrame({
            "Required Column": list(auto_mapping.keys()),
            "Your Column": [auto_mapping[col] for col in auto_mapping.keys()]
        })
        st.dataframe(mapping_df, use_container_width=True)

        st.markdown("#### Manual Mapping")
        st.info("If the auto-mapping is incorrect, please manually map your columns below.")

        manual_mapping = {}
        for required_col in REQUIRED_SERVICE_COLUMNS:
            options = [""] + user_columns
            default_idx = 0
            if auto_mapping[required_col] in user_columns:
                default_idx = options.index(auto_mapping[required_col])
            manual_mapping[required_col] = st.selectbox(
                f"Map **{required_col}** to:",
                options=options,
                index=default_idx,
                key=f"map_{required_col}"
            )

        st.session_state.column_mapping = {k: v for k, v in manual_mapping.items() if v}

        if st.button("Save Column Mapping"):
            st.success("✅ Column mapping saved!")

def cleanup_temp_files():
    """Clean up temporary files on app exit."""
    for temp_file in ['service_temp_file', 'drug_temp_file', 'raw_data_file']:
        if temp_file in st.session_state and os.path.exists(st.session_state[temp_file]):
            if not try_delete_file(st.session_state[temp_file]):
                st.warning(f"Could not delete {st.session_state[temp_file]}. It may be in use.")
with tab3:
    st.markdown('<div class="step-header">Step 3: Configure & Process Claims</div>', unsafe_allow_html=True)

    if 'raw_data' not in st.session_state or 'column_mapping' not in st.session_state:
        st.warning("⚠️ Please upload a file and map columns in the previous tabs.")
    else:
        df = st.session_state.raw_data
        mapping = st.session_state.column_mapping

        # Rename columns according to mapping
        renamed_df = df.rename(columns={v: k for k, v in mapping.items() if v})

        # Filter data into services and drugs
        medicine_keywords = ['medicine', 'drug', 'pharmacy', 'medication']
        if 'service_type' in renamed_df.columns:
            drug_mask = renamed_df['service_type'].str.lower().str.contains('|'.join(medicine_keywords), na=False)
            st.session_state.filtered_drug_data = renamed_df[drug_mask].copy()
            st.session_state.filtered_service_data = renamed_df[~drug_mask].copy()
        else:
            st.session_state.filtered_service_data = renamed_df.copy()
            st.session_state.filtered_drug_data = pd.DataFrame(columns=renamed_df.columns)

        # Show what will be processed
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"""
        **Ready to Process:**
        - Provider: `{st.session_state.get('selected_provider', 'N/A')}`
        - Raw Data: `{Path(st.session_state.raw_data_file).name}`
        - <span class="service-badge">Services: {len(st.session_state.filtered_service_data):,}</span>
        - <span class="drug-badge">Drugs: {len(st.session_state.filtered_drug_data):,}</span>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Processing mode selection
        st.markdown("#### Processing Mode")

        if len(st.session_state.filtered_service_data) > 0 and len(st.session_state.filtered_drug_data) > 0:
            processing_options = ['Automatic (Both)', 'Services Only', 'Drugs Only']
        elif len(st.session_state.filtered_service_data) > 0:
            processing_options = ['Services Only']
        elif len(st.session_state.filtered_drug_data) > 0:
            processing_options = ['Drugs Only']
        else:
            st.error("❌ No valid claims detected in uploaded data")
            processing_options = []

        if processing_options:
            processing_mode = st.radio(
                "Select processing mode:",
                options=processing_options,
                horizontal=True
            )

            st.session_state.processing_mode = processing_mode

        st.markdown("#### Processing Parameters")
        col1, col2 = st.columns(2)
        with col1:
            semantic_threshold = st.slider(
                "Semantic Matching Threshold (Services)",
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
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            if not openai_api_key:
                st.error("❌ OpenAI API Key is required for claim justification")
            else:
                # Create output directory
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                
                status_placeholder = st.empty()
                progress_container = st.container()

                try:
                    mode = st.session_state.processing_mode
                    process_services_flag = 'Services' in mode or 'Automatic' in mode
                    process_drugs_flag = 'Drugs' in mode or 'Automatic' in mode

                    # Check if we have necessary reference files
                    has_service_refs = all([
                        hasattr(st.session_state, 'selected_provider'),
                        hasattr(st.session_state, 'pricing_file'),
                        hasattr(st.session_state, 'naphies_file')
                    ])

                    has_drug_refs = all([
                        hasattr(st.session_state, 'sfda_file'),
                        hasattr(st.session_state, 'chi_file')
                    ])

                    if process_services_flag and not has_service_refs:
                        st.error("❌ Missing service reference files (Pricing/NAPHIES)")
                        st.stop()

                    if process_drugs_flag and not has_drug_refs:
                        st.error("❌ Missing drug reference files (SFDA/CHI)")
                        st.stop()

                    results = {}

                    # Save filtered data to temp files
                    if process_services_flag:
                        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
                            service_temp_path = tmp.name
                            save_to_temp_file(st.session_state.filtered_service_data, service_temp_path)
                        st.session_state.service_temp_file = service_temp_path

                    if process_drugs_flag:
                        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
                            drug_temp_path = tmp.name
                            save_to_temp_file(st.session_state.filtered_drug_data, drug_temp_path)
                        st.session_state.drug_temp_file = drug_temp_path

                    # Process in parallel if both types
                    if process_services_flag and process_drugs_flag:
                        status_placeholder.info("🔄 Processing Services and Drugs in parallel...")
                        
                        with progress_container:
                            # Create separate status placeholders for each process
                            service_status = st.empty()
                            drug_status = st.empty()

                                        # NEW CODE (put this in the same place):
                    if process_services_flag and process_drugs_flag:
                        status_placeholder.info("🔄 Processing Services...")
                        results['services'] = process_services_with_progress(
                            st.session_state.service_temp_file,
                            st.session_state.pricing_file,
                            st.session_state.naphies_file,
                            semantic_threshold,
                            openai_api_key,
                            batch_size,
                            max_workers,
                            status_placeholder,  # Use the same placeholder for both
                            st.session_state.get('selected_sheet', 'Worksheet')
                        )
                        status_placeholder.info("🔄 Processing Drugs...")
                        results['drugs'] = process_drugs_with_progress(
                            st.session_state.drug_temp_file,
                            st.session_state.sfda_file,
                            st.session_state.chi_file,
                            openai_api_key,
                            batch_size,
                            max_workers,
                            status_placeholder,  # Use the same placeholder for both
                            st.session_state.get('selected_sheet', 'Worksheet')
                        )
                    elif process_services_flag:
                        status_placeholder.info("🔄 Processing Services...")
                        results['services'] = process_services_with_progress(...)
                    elif process_drugs_flag:
                        status_placeholder.info("🔄 Processing Drugs...")
                        results['drugs'] = process_drugs_with_progress(...)

                    # Process drugs only
                    elif process_drugs_flag:
                        with progress_container:
                            drug_status = st.empty()
                        
                        results['drugs'] = process_drugs_with_progress(
                            st.session_state.drug_temp_file,
                            st.session_state.sfda_file,
                            st.session_state.chi_file,
                            openai_api_key,
                            batch_size,
                            max_workers,
                            drug_status,
                            st.session_state.get('selected_sheet', 'Worksheet')
                        )

                    # Store results in session state ONLY after completion
                    if 'services' in results and results['services']['status'] == 'success':
                        st.session_state.service_matched_results = results['services']['matched']
                        st.session_state.service_justified_results = results['services']['justified']

                    if 'drugs' in results and results['drugs']['status'] == 'success':
                        st.session_state.drug_matched_results = results['drugs']['matched']
                        st.session_state.drug_justified_results = results['drugs']['justified']

                    # Only mark as complete if at least one processing succeeded
                    if any(r.get('status') == 'success' for r in results.values()):
                        st.session_state.processing_complete = True
                        st.balloons()
                        status_placeholder.success("🎉 **Processing Complete!** Check the Results tab for details.")
                    else:
                        status_placeholder.error("❌ All processing tasks failed. Check the error messages above.")

                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)

                finally:
                    # Give more time for file handles to release
                    time.sleep(2)
                    
                    # Cleanup temp files silently
                    for temp_key in ['service_temp_file', 'drug_temp_file', 'raw_data_file']:
                        if temp_key in st.session_state:
                            try_delete_file(st.session_state.get(temp_key, ''), retries=5, delay=1)

with tab4:
    st.markdown('<div class="step-header">Step 4: View Results</div>', unsafe_allow_html=True)
    if not st.session_state.processing_complete:
        st.info("⏳ No results yet. Please complete processing in the Processing tab.")
    else:
        # Create sub-tabs for services and drugs
        result_tabs = []
        if st.session_state.service_justified_results is not None:
            result_tabs.append("📋 Services")
        if st.session_state.drug_justified_results is not None:
            result_tabs.append("💊 Drugs")

        if result_tabs:
            result_tab_objects = st.tabs(result_tabs)
            tab_idx = 0

            # Services Results Tab
            if st.session_state.service_justified_results is not None:
                with result_tab_objects[tab_idx]:
                    results = st.session_state.service_justified_results

                    st.markdown("#### 📊 Service Claims Summary")
                    col1, col2, col3, col4 = st.columns(4)

                    total_claims = len(results)
                    path1_count = (results['match_path'] == 'Path1_Exact').sum()
                    path2_count = (results['match_path'] == 'Path2_Matched').sum()
                    justified_count = (results['status'] == 'Justified').sum()
                    needs_review_count = results['needs_manual_review'].sum()

                    with col1:
                        st.metric("Total Services", f"{total_claims:,}")
                    with col2:
                        st.metric("Matched", f"{path1_count + path2_count:,}",
                                 f"{((path1_count + path2_count)/total_claims*100):.1f}%")
                    with col3:
                        st.metric("Justified", f"{justified_count:,}",
                                 f"{(justified_count/total_claims*100):.1f}%")
                    with col4:
                        st.metric("Needs Review", f"{needs_review_count:,}",
                                 f"{(needs_review_count/total_claims*100):.1f}%")

                    # Detailed table
                    with st.expander("🔍 Detailed Service Results"):
                        display_cols = [
                            'needs_manual_review', 'status', 'match_path', 'match_score',
                            'naphies_code', 'service_description', 'Diagnoses', 'note'
                        ]
                        st.dataframe(results[display_cols].head(100), use_container_width=True)

                tab_idx += 1

            # Drugs Results Tab
            if st.session_state.drug_justified_results is not None:
                with result_tab_objects[tab_idx]:
                    results = st.session_state.drug_justified_results

                    st.markdown("#### 💊 Drug Claims Summary")
                    col1, col2, col3, col4 = st.columns(4)

                    total_claims = len(results)
                    chi_matched = (results['match_path'] == 'CHI_Matched').sum()
                    justified_count = (results['status'] == 'Justified').sum()
                    needs_review_count = results['needs_manual_review'].sum()

                    with col1:
                        st.metric("Total Drugs", f"{total_claims:,}")
                    with col2:
                        st.metric("CHI Matched", f"{chi_matched:,}",
                                 f"{(chi_matched/total_claims*100):.1f}%")
                    with col3:
                        st.metric("Justified", f"{justified_count:,}",
                                 f"{(justified_count/total_claims*100):.1f}%")
                    with col4:
                        st.metric("Needs Review", f"{needs_review_count:,}",
                                 f"{(needs_review_count/total_claims*100):.1f}%")

                    # Detailed table
                    with st.expander("🔍 Detailed Drug Results"):
                        display_cols = [
                            'needs_manual_review', 'status', 'match_path',
                            'scientific_name', 'chi_icd10_code', 'chi_indication',
                            'service_description', 'Diagnoses', 'note'
                        ]
                        st.dataframe(results[display_cols].head(100), use_container_width=True)

        st.markdown("---")
        st.info("📥 Download all processed files in the **Download** tab.")

with tab5:
    st.markdown('<div class="step-header">Step 5: Download Results</div>', unsafe_allow_html=True)
    if not st.session_state.processing_complete:
        st.info("⏳ No results available for download. Please complete processing first.")
    else:
        st.markdown("#### 📥 Download Processed Files")
        provider_name = st.session_state.get('selected_provider', 'unknown')

        # Services Downloads
        if st.session_state.service_justified_results is not None:
            st.markdown("##### <span class='service-badge'>SERVICE CLAIMS</span>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)

            with col1:
                if os.path.exists('output/service_justified_claims.xlsx'):
                    with open('output/service_justified_claims.xlsx', 'rb') as f:
                        st.download_button(
                            label="📄 All Service Results",
                            data=f,
                            file_name=f"{provider_name}_services_justified.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

            with col2:
                if os.path.exists('output/service_no_match_claims.xlsx'):
                    with open('output/service_no_match_claims.xlsx', 'rb') as f:
                        st.download_button(
                            label="❌ Service No-Match",
                            data=f,
                            file_name=f"{provider_name}_services_no_match.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

            with col3:
                # Use BytesIO for review required
                review_claims = st.session_state.service_justified_results[
                    st.session_state.service_justified_results['needs_manual_review'] == True
                ]
                if not review_claims.empty:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        review_claims.to_excel(writer, index=False, sheet_name='Review_Required')
                    output.seek(0)
                    st.download_button(
                        label=f"⚠️ Service Review Required ({len(review_claims)} claims)",
                        data=output,
                        file_name=f"{provider_name}_services_review.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                else:
                    st.info("✅ No service claims require manual review")

        st.markdown("---")

        # Drugs Downloads
        if st.session_state.drug_justified_results is not None:
            st.markdown("##### <span class='drug-badge'>DRUG CLAIMS</span>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)

            with col1:
                if os.path.exists('output/drug_justified_claims.xlsx'):
                    with open('output/drug_justified_claims.xlsx', 'rb') as f:
                        st.download_button(
                            label="📄 All Drug Results",
                            data=f,
                            file_name=f"{provider_name}_drugs_justified.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

            with col2:
                if os.path.exists('output/drug_no_match_claims.xlsx'):
                    with open('output/drug_no_match_claims.xlsx', 'rb') as f:
                        st.download_button(
                            label="❌ Drug No-Match",
                            data=f,
                            file_name=f"{provider_name}_drugs_no_match.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

            with col3:
                # Use BytesIO for review required
                review_claims = st.session_state.drug_justified_results[
                    st.session_state.drug_justified_results['needs_manual_review'] == True
                ]
                if not review_claims.empty:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        review_claims.to_excel(writer, index=False, sheet_name='Review_Required')
                    output.seek(0)
                    st.download_button(
                        label=f"⚠️ Drug Review Required ({len(review_claims)} claims)",
                        data=output,
                        file_name=f"{provider_name}_drugs_review.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                else:
                    st.info("✅ No drug claims require manual review")

        st.markdown("---")
        st.markdown("#### 📋 File Descriptions")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Service Claims Files:**")
            st.markdown("""
            - **All Results**: Complete dataset with NAPHIES matching & AI justification
            - **No-Match**: Claims without NAPHIES code match
            - **Review Required**: Claims flagged for manual review by AI
            """)

        with col2:
            st.markdown("**Drug Claims Files:**")
            st.markdown("""
            - **All Results**: Complete dataset with SFDA/CHI matching & AI justification
            - **No-Match**: Claims without SFDA/CHI match
            - **Review Required**: Claims flagged for manual review by AI
            """)

        st.markdown("---")
        if st.button("🔄 Reset & Start New Batch", type="secondary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Healthcare Claims Processing System v2.0 | Unified Services & Drugs Pipeline | Powered by AI 🤖"
    "</div>",
    unsafe_allow_html=True
)
