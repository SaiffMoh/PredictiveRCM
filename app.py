from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import io

# Load environment variables
load_dotenv()

# Import the claim processing classes
from claim_matcher import ClaimMatcher
from claim_justifier import ClaimJustifier
from drug_matcher import DrugMatcher
from drug_justifier import DrugJustifier

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

def process_services(raw_file, pricing_file, naphies_file, semantic_threshold, openai_api_key, batch_size, max_workers):
    """Process service claims"""
    try:
        # Service Matching
        matcher = ClaimMatcher()
        matched_results = matcher.run_pipeline(
            raw_file=raw_file,
            pricing_file=pricing_file,
            naphies_file=naphies_file,
            output_file='output/service_matched_claims.xlsx',
            no_match_file='output/service_no_match_claims.xlsx',
            review_file='output/service_needs_review_claims.xlsx',
            semantic_threshold=semantic_threshold
        )
        
        # Service Justification
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
        
        return {'matched': matched_results, 'justified': justified_results, 'status': 'success'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def process_drugs(raw_file, sfda_file, chi_file, openai_api_key, batch_size, max_workers):
    """Process drug claims"""
    try:
        # Drug Matching
        matcher = DrugMatcher()
        matched_results = matcher.run_pipeline(
            raw_file=raw_file,
            sfda_file=sfda_file,
            chi_file=chi_file,
            output_file='output/drug_matched_claims.xlsx',
            no_match_file='output/drug_no_match_claims.xlsx',
            review_file='output/drug_needs_review_claims.xlsx'
        )
        
        # Drug Justification
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
        
        return {'matched': matched_results, 'justified': justified_results, 'status': 'success'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

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
    3. **Configure Settings** - Set parameters
    4. **Process Claims** - Run matching & justification
    5. **Download Results** - Export processed data
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
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.raw_data_file = temp_path
        st.success(f"✓ Uploaded: {uploaded_file.name}")

        # Detect claim types and show summary
        try:
            preview_df = pd.read_excel(temp_path, sheet_name='Worksheet')
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

with tab2:
    st.markdown('<div class="step-header">Step 2: Configure & Process Claims</div>', unsafe_allow_html=True)

    # Check if all required files are available
    has_service_refs = all([
        hasattr(st.session_state, 'selected_provider'),
        hasattr(st.session_state, 'pricing_file'),
        hasattr(st.session_state, 'naphies_file')
    ])
    
    has_drug_refs = all([
        hasattr(st.session_state, 'sfda_file'),
        hasattr(st.session_state, 'chi_file')
    ])
    
    has_raw_data = hasattr(st.session_state, 'raw_data_file')

    if not has_raw_data:
        st.warning("⚠️ Please upload raw claims data in Step 1 (Setup)")
    else:
        # Show what will be processed
        if st.session_state.claim_types:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(f"""
            **Ready to Process:**
            - Provider: `{st.session_state.get('selected_provider', 'N/A')}`
            - Raw Data: `{Path(st.session_state.raw_data_file).name}`
            - <span class="service-badge">Services: {st.session_state.claim_types['services']:,}</span>
            - <span class="drug-badge">Drugs: {st.session_state.claim_types['drugs']:,}</span>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Processing mode selection
        st.markdown("#### Processing Mode")
        
        if st.session_state.claim_types:
            if st.session_state.claim_types['services'] > 0 and st.session_state.claim_types['drugs'] > 0:
                processing_options = ['Automatic (Both)', 'Services Only', 'Drugs Only']
            elif st.session_state.claim_types['services'] > 0:
                processing_options = ['Services Only']
            elif st.session_state.claim_types['drugs'] > 0:
                processing_options = ['Drugs Only']
            else:
                st.error("❌ No valid claims detected in uploaded data")
                processing_options = []
        else:
            processing_options = ['Automatic (Both)']
        
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
            elif not has_raw_data:
                st.error("❌ No raw data uploaded")
            else:
                # Create output directory
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)

                progress_placeholder = st.empty()
                status_placeholder = st.empty()

                try:
                    mode = st.session_state.processing_mode
                    process_services_flag = 'Services' in mode or 'Automatic' in mode
                    process_drugs_flag = 'Drugs' in mode or 'Automatic' in mode
                    
                    # Check if we have necessary reference files
                    if process_services_flag and not has_service_refs:
                        st.error("❌ Missing service reference files (Pricing/NAPHIES)")
                        st.rerun()
                    
                    if process_drugs_flag and not has_drug_refs:
                        st.error("❌ Missing drug reference files (SFDA/CHI)")
                        st.rerun()
                    
                    results = {}
                    
                    # Process in parallel if both types
                    if process_services_flag and process_drugs_flag:
                        status_placeholder.info("🔄 Processing Services and Drugs in parallel...")
                        
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            futures = {}
                            
                            # Submit service processing
                            futures['services'] = executor.submit(
                                process_services,
                                st.session_state.raw_data_file,
                                st.session_state.pricing_file,
                                st.session_state.naphies_file,
                                semantic_threshold,
                                openai_api_key,
                                batch_size,
                                max_workers
                            )
                            
                            # Submit drug processing
                            futures['drugs'] = executor.submit(
                                process_drugs,
                                st.session_state.raw_data_file,
                                st.session_state.sfda_file,
                                st.session_state.chi_file,
                                openai_api_key,
                                batch_size,
                                max_workers
                            )
                            
                            # Collect results
                            for claim_type, future in futures.items():
                                result = future.result()
                                results[claim_type] = result
                                if result['status'] == 'success':
                                    status_placeholder.success(f"✅ {claim_type.title()} processing complete!")
                                else:
                                    status_placeholder.error(f"❌ {claim_type.title()} processing failed: {result['error']}")
                    
                    # Process services only
                    elif process_services_flag:
                        status_placeholder.info("🔄 Processing Service Claims...")
                        results['services'] = process_services(
                            st.session_state.raw_data_file,
                            st.session_state.pricing_file,
                            st.session_state.naphies_file,
                            semantic_threshold,
                            openai_api_key,
                            batch_size,
                            max_workers
                        )
                    
                    # Process drugs only
                    elif process_drugs_flag:
                        status_placeholder.info("🔄 Processing Drug Claims...")
                        results['drugs'] = process_drugs(
                            st.session_state.raw_data_file,
                            st.session_state.sfda_file,
                            st.session_state.chi_file,
                            openai_api_key,
                            batch_size,
                            max_workers
                        )
                    
                    # Store results in session state
                    if 'services' in results and results['services']['status'] == 'success':
                        st.session_state.service_matched_results = results['services']['matched']
                        st.session_state.service_justified_results = results['services']['justified']
                    
                    if 'drugs' in results and results['drugs']['status'] == 'success':
                        st.session_state.drug_matched_results = results['drugs']['matched']
                        st.session_state.drug_justified_results = results['drugs']['justified']
                    
                    st.session_state.processing_complete = True
                    st.balloons()
                    status_placeholder.success("🎉 **Processing Complete!** Check the Results tab for details.")

                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)
                finally:
                    # Cleanup temp file
                    if 'raw_data_file' in st.session_state and os.path.exists(st.session_state.raw_data_file):
                        os.remove(st.session_state.raw_data_file)

with tab3:
    st.markdown('<div class="step-header">Step 3: View Results</div>', unsafe_allow_html=True)

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

with tab4:
    st.markdown('<div class="step-header">Step 4: Download Results</div>', unsafe_allow_html=True)

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