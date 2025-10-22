import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class ClaimMatcher:
    def __init__(self):
        """Initialize the matcher"""
        self.raw_data = None
        self.pricing_list = None
        self.naphies_ref = None
        self.results = None

        # Initialize sentence transformer for semantic matching
        print("Loading sentence transformer model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully!\n")

    def load_data(self, raw_file_path, pricing_file_path, naphies_file_path):
        """
        Load all required data files
        """
        print("="*60)
        print("LOADING DATA FILES")
        print("="*60)

        self.raw_data = pd.read_excel(raw_file_path, sheet_name='Worksheet')
        # Normalize service_code to string
        self.raw_data['service_code'] = self.raw_data['service_code'].fillna('').astype(str)
        print(f"✓ Loaded {len(self.raw_data):,} rows from raw data")
        print(f"→ Duplicate rows in raw data: {self.raw_data.duplicated().sum()}")

        self.pricing_list = pd.read_excel(pricing_file_path, sheet_name='Worksheet')
        # Normalize payer_code to string
        self.pricing_list['payer_code'] = self.pricing_list['payer_code'].fillna('').astype(str)
        print(f"✓ Loaded {len(self.pricing_list):,} rows from pricing list")

        self.naphies_ref = pd.read_excel(naphies_file_path, sheet_name='SBS V2.0 Tabular List ')
        print(f"✓ Loaded {len(self.naphies_ref):,} rows from NAPHIES reference\n")

        self._display_data_info()

    def _display_data_info(self):
        """Display information about loaded data"""
        print("="*60)
        print("DATA OVERVIEW")
        print("="*60)

        print("\nRaw Data Columns:")
        print(self.raw_data.columns.tolist())

        print("\nPricing List Columns:")
        print(self.pricing_list.columns.tolist())

        print("\nNAPHIES Reference Columns:")
        print(self.naphies_ref.columns.tolist())
        print("\n")

    def filter_non_medicines(self):
        """
        Filter raw data to only include non-medicine claims
        """
        print("="*60)
        print("FILTERING NON-MEDICINE CLAIMS")
        print("="*60)

        initial_count = len(self.raw_data)

        if 'service_type' in self.raw_data.columns:
            medicine_keywords = ['medicine', 'drug', 'pharmacy', 'medication']
            mask = ~self.raw_data['service_type'].str.lower().str.contains(
                '|'.join(medicine_keywords),
                na=False
            )
            self.raw_data = self.raw_data[mask].copy()

        filtered_count = len(self.raw_data)
        print(f"Initial rows: {initial_count:,}")
        print(f"After filtering: {filtered_count:,}")
        print(f"Removed: {initial_count - filtered_count:,} medicine claims\n")

    def exact_match(self):
        """
        Path 1: Perform exact matching using Company + Service Code
        Handles empty company or service_code by routing to Path 2
        """
        print("="*60)
        print("PATH 1: EXACT MATCHING")
        print("="*60)

        # Initialize needs_manual_review column
        self.raw_data['needs_manual_review'] = False

        # Handle missing company or service_code
        missing_key_mask = self.raw_data['company'].isna() | self.raw_data['service_code'].isna()
        self.raw_data.loc[missing_key_mask, 'match_path'] = 'Path2_Pending'
        self.raw_data.loc[missing_key_mask, 'mismatch_reason'] = 'Missing company or service code in raw data'
        self.raw_data.loc[missing_key_mask, 'needs_manual_review'] = True

        # Create matching keys for non-missing rows
        non_missing = ~missing_key_mask
        self.raw_data.loc[non_missing, 'match_key'] = (
            self.raw_data.loc[non_missing, 'company'].astype(str).str.strip().str.lower() + '_' +
            self.raw_data.loc[non_missing, 'service_code'].astype(str).str.strip()
        )

        # Normalize naphies_code in pricing list BEFORE creating match_key
        self.pricing_list['naphies_code'] = self.pricing_list['naphies_code'].astype(str).str.strip().str.upper()
        self.pricing_list.loc[self.pricing_list['naphies_code'] == 'NAN', 'naphies_code'] = np.nan

        self.pricing_list['match_key'] = (
            self.pricing_list['company_name'].astype(str).str.strip().str.lower() + '_' +
            self.pricing_list['payer_code'].astype(str).str.strip()
        )

        # Perform exact match on non-missing
        matched = self.raw_data[non_missing].merge(
            self.pricing_list[['match_key', 'naphies_code']],
            on='match_key',
            how='left'
        )

        # Check if match_key exists in pricing list
        matched['in_pricing_list'] = matched['match_key'].isin(self.pricing_list['match_key'])

        # Identify Path 1 matches (where naphies_code exists)
        path1_mask = matched['naphies_code'].notna()
        matched['match_path'] = np.where(path1_mask, 'Path1_Exact', 'Path2_Pending')
        matched['match_score'] = np.where(path1_mask, 1.0, np.nan)
        matched['needs_manual_review'] = ~path1_mask  # Path 1 exact matches don't need review
        matched['mismatch_reason'] = np.where(
            path1_mask,
            None,
            np.where(matched['in_pricing_list'],
                     'Missing naphies_code in pricing list',
                     'Company + Service Code not in pricing list')
        )

        # Combine back with missing key rows
        self.results = pd.concat([matched, self.raw_data[missing_key_mask]], ignore_index=True)

        path1_count = (self.results['match_path'] == 'Path1_Exact').sum()
        path1_pct = (path1_count / len(self.results)) * 100 if len(self.results) > 0 else 0
        pending_count = (self.results['match_path'] == 'Path2_Pending').sum()
        missing_raw_count = missing_key_mask.sum()

        print(f"✓ Path 1 matches: {path1_count:,} / {len(self.results):,} ({path1_pct:.1f}%)")
        print(f"→ Remaining for Path 2: {pending_count:,}")
        print(f"→ Path 2 due to missing naphies_code: "
              f"{((self.results['match_path'] == 'Path2_Pending') & self.results['in_pricing_list'].fillna(False)).sum():,}")
        print(f"→ Path 2 due to missing company/service code: "
              f"{((self.results['match_path'] == 'Path2_Pending') & ~self.results['in_pricing_list'].fillna(False)).sum():,}")
        print(f"→ Path 2 due to missing raw company/service code: {missing_raw_count:,}\n")

        if path1_count > 0:
            sample_codes = self.results[self.results['match_path'] == 'Path1_Exact']['naphies_code'].head(5)
            print(f"Sample naphies_codes from Path 1: {sample_codes.tolist()}\n")

        # Export missing match_keys for review
        missing_keys = set(self.results[self.results['match_path'] == 'Path2_Pending']['match_key'].dropna())
        if missing_keys:
            pd.DataFrame(list(missing_keys), columns=['missing_match_key']).to_excel(
                './output/missing_pricing_keys.xlsx', index=False)
            print(f"Exported {len(missing_keys):,} missing match keys to './output/missing_pricing_keys.xlsx'\n")

        return self.results

    def semantic_match(self, threshold=0.75, top_k=3):
        """
        Path 2: Semantic matching for rows without exact match
        ALL Path 2 matches are flagged for manual review
        """
        print("="*60)
        print("PATH 2: SEMANTIC MATCHING")
        print("="*60)

        path2_mask = self.results['match_path'] == 'Path2_Pending'
        path2_data = self.results[path2_mask].copy()

        if len(path2_data) == 0:
            print("No rows requiring semantic matching.\n")
            return self.results

        print(f"Processing {len(path2_data):,} rows with semantic matching...")

        # Prepare NAPHIES texts
        naphies_texts = []
        naphies_codes = []
        for idx, row in self.naphies_ref.iterrows():
            text_parts = []
            if pd.notna(row.get('Short Description')):
                text_parts.append(str(row['Short Description']))
            if pd.notna(row.get('Long Description')):
                text_parts.append(str(row['Long Description']))
            if pd.notna(row.get('Includes')):
                text_parts.append(str(row['Includes']))
            naphies_texts.append(' '.join(text_parts) if text_parts else '')
            naphies_codes.append(str(row.get('SBS Code (Hyphenated) ', '')).strip().upper())

        print("Generating NAPHIES embeddings...")
        naphies_embeddings = self.embedder.encode(naphies_texts, show_progress_bar=True)

        # Prepare raw texts
        raw_texts = []
        for idx, row in path2_data.iterrows():
            parts = []
            if pd.notna(row.get('service_description')):
                parts.append(str(row['service_description']))
            if pd.notna(row.get('Diagnoses')):
                parts.append(str(row['Diagnoses']))
            if pd.notna(row.get('chief_complaint')):
                parts.append(str(row['chief_complaint']))
            raw_texts.append(' '.join(parts) if parts else '')

        print("Generating embeddings for raw descriptions...")
        raw_embeddings = self.embedder.encode(raw_texts, show_progress_bar=True)

        print("Computing similarity scores...")
        similarities = cosine_similarity(raw_embeddings, naphies_embeddings)

        # Find top matches
        matched_naphies = []
        match_scores = []
        match_status = []
        top_candidates = []
        mismatch_reasons = []

        for i, sim_scores in enumerate(similarities):
            top_indices = np.argsort(sim_scores)[-top_k:][::-1]
            top_scores = sim_scores[top_indices]
            best_idx = top_indices[0]
            best_score = top_scores[0]

            candidates = [
                {'naphies_code': naphies_codes[idx], 'score': float(score),
                 'description': naphies_texts[idx][:100]}
                for idx, score in zip(top_indices, top_scores)
            ]
            top_candidates.append(candidates)

            if best_score >= threshold:
                matched_naphies.append(naphies_codes[best_idx])
                match_scores.append(best_score)
                match_status.append('Path2_Matched')
                mismatch_reasons.append(None)
            else:
                matched_naphies.append(None)
                match_scores.append(best_score)
                match_status.append('No_Match')
                existing_reason = path2_data.iloc[i]['mismatch_reason']
                mismatch_reasons.append(
                    f"{existing_reason}; Low similarity score ({best_score:.3f} < {threshold})"
                )

        # Update results
        path2_indices = self.results[path2_mask].index
        self.results.loc[path2_indices, 'naphies_code'] = matched_naphies
        self.results.loc[path2_indices, 'match_score'] = match_scores
        self.results.loc[path2_indices, 'match_path'] = match_status
        self.results.loc[path2_indices, 'top_candidates'] = [str(c) for c in top_candidates]
        self.results.loc[path2_indices, 'mismatch_reason'] = mismatch_reasons
        # CRITICAL: All Path2 matches and No_Match need manual review
        self.results.loc[path2_indices, 'needs_manual_review'] = True

        path2_matched = sum(1 for s in match_status if s == 'Path2_Matched')
        no_match = sum(1 for s in match_status if s == 'No_Match')
        path2_pct = (path2_matched / len(path2_data)) * 100 if len(path2_data) > 0 else 0

        print(f"✓ Path 2 matches: {path2_matched:,} / {len(path2_data):,} ({path2_pct:.1f}%)")
        print(f"✗ No match found: {no_match:,} ({(no_match/len(path2_data)*100):.1f}%)")
        print(f"⚠ ALL Path 2 matches and No_Match flagged for manual review\n")

        return self.results

    def enrich_with_naphies_data(self):
        """
        Enrich matched results with NAPHIES reference data
        Works for BOTH Path 1 and Path 2 matches
        """
        print("="*60)
        print("ENRICHING WITH NAPHIES DATA")
        print("="*60)

        # Normalize BOTH columns properly before merge
        self.naphies_ref['SBS Code (Hyphenated) '] = self.naphies_ref['SBS Code (Hyphenated) '].astype(str).str.strip().str.upper()
        self.results['naphies_code'] = self.results['naphies_code'].astype(str).str.strip().str.upper()

        # Remove 'NAN' strings
        self.naphies_ref.loc[self.naphies_ref['SBS Code (Hyphenated) '] == 'NAN', 'SBS Code (Hyphenated) '] = np.nan
        self.results.loc[self.results['naphies_code'] == 'NAN', 'naphies_code'] = np.nan

        # Debug - check for matches before merge
        matched_codes_count = self.results['naphies_code'].isin(self.naphies_ref['SBS Code (Hyphenated) ']).sum()
        print(f"→ Codes found in NAPHIES reference: {matched_codes_count:,} / {self.results['naphies_code'].notna().sum():,}")

        # Sample some codes
        sample_result_codes = self.results[self.results['naphies_code'].notna()]['naphies_code'].head(3).tolist()
        sample_naphies_codes = self.naphies_ref['SBS Code (Hyphenated) '].head(3).tolist()
        print(f"→ Sample result codes: {sample_result_codes}")
        print(f"→ Sample NAPHIES codes: {sample_naphies_codes}")

        # Check overlap
        result_codes_set = set(self.results[self.results['naphies_code'].notna()]['naphies_code'].unique())
        naphies_codes_set = set(self.naphies_ref[self.naphies_ref['SBS Code (Hyphenated) '].notna()]['SBS Code (Hyphenated) '].unique())
        overlap = result_codes_set.intersection(naphies_codes_set)
        print(f"→ Overlapping codes: {len(overlap):,} / {len(result_codes_set):,}\n")

        # Select relevant NAPHIES columns
        naphies_cols = ['SBS Code', 'SBS Code (Hyphenated) ', 'Short Description', 'Long Description',
                       'Chapter', 'Chapter Name', 'Service Category', 'Limit per Day',
                       'Block', 'Block Name', 'Definition', 'Includes', 'Excludes ', 'Guidelines']

        # Merge using 'SBS Code (Hyphenated) ' column
        self.results = self.results.merge(
            self.naphies_ref[naphies_cols],
            left_on='naphies_code',
            right_on='SBS Code (Hyphenated) ',
            how='left',
            suffixes=('', '_naphies'),
            indicator=True
        )

        # Check merge results
        merge_success = (self.results['_merge'] == 'both').sum()
        merge_left_only = (self.results['_merge'] == 'left_only').sum()
        print(f"→ Successful merges: {merge_success:,}")
        print(f"→ Failed merges: {merge_left_only:,}")

        # Drop merge indicator
        self.results = self.results.drop('_merge', axis=1)

        # Count enrichment by path
        path1_enriched = self.results[
            (self.results['match_path'] == 'Path1_Exact') &
            (self.results['Short Description'].notna())
        ].shape[0]
        path2_enriched = self.results[
            (self.results['match_path'] == 'Path2_Matched') &
            (self.results['Short Description'].notna())
        ].shape[0]

        print(f"✓ Path 1 enriched: {path1_enriched:,}")
        print(f"✓ Path 2 enriched: {path2_enriched:,}")
        print(f"✓ Total enriched: {path1_enriched + path2_enriched:,}\n")

        # Verify enrichment
        if path1_enriched + path2_enriched > 0:
            sample_row = self.results[self.results['Short Description'].notna()].iloc[0]
            print(f"Sample enriched row:")
            print(f"  → Match Path: {sample_row['match_path']}")
            print(f"  → NAPHIES Code: {sample_row['naphies_code']}")
            print(f"  → Short Description: {sample_row.get('Short Description', 'N/A')[:80]}...")
            print(f"  → Needs Review: {sample_row['needs_manual_review']}\n")
        else:
            print("WARNING: No rows were successfully enriched!\n")

        return self.results

    def generate_summary(self):
        """Generate and display summary statistics"""
        print("="*60)
        print("PROCESSING SUMMARY")
        print("="*60)

        total = len(self.results)
        path1 = (self.results['match_path'] == 'Path1_Exact').sum()
        path2 = (self.results['match_path'] == 'Path2_Matched').sum()
        no_match = (self.results['match_path'] == 'No_Match').sum()

        # Enrichment check
        enriched = self.results['Short Description'].notna().sum()

        # Manual review check
        needs_review = self.results['needs_manual_review'].sum()

        print(f"\nTotal rows processed: {total:,}")

        print(f"\nPath 1 (Exact Match): {path1:,} ({path1/total*100:.1f}%)")
        print(f"Path 2 (Semantic Match): {path2:,} ({path2/total*100:.1f}%)")
        print(f"No Match: {no_match:,} ({no_match/total*100:.1f}%)")

        print(f"\nSuccessfully Enriched: {enriched:,} / {path1+path2:,} ({enriched/(path1+path2)*100 if path1+path2>0 else 0:.1f}% of matched)")
        print(f"\n⚠ NEEDS MANUAL REVIEW: {needs_review:,} ({needs_review/total*100:.1f}%)")
        print(f"  → Path 2 matches: {(self.results['match_path'] == 'Path2_Matched').sum():,} (semantic matching)")
        print(f"  → No Match: {no_match:,} (no NAPHIES code found)")

        # Breakdown of no-match reasons
        if no_match > 0:
            print("\nNo-Match Breakdown:")
            no_match_rows = self.results[self.results['match_path'] == 'No_Match']
            missing_pricing = no_match_rows['mismatch_reason'].str.contains(
                'Company + Service Code not in pricing list', na=False).sum()
            missing_naphies = no_match_rows['mismatch_reason'].str.contains(
                'Missing naphies_code in pricing list', na=False).sum()
            low_similarity = no_match_rows['mismatch_reason'].str.contains(
                'Low similarity', na=False).sum()
            missing_raw = no_match_rows['mismatch_reason'].str.contains(
                'Missing company or service code in raw data', na=False).sum()
            print(f"  → Not in pricing list: {missing_pricing:,}")
            print(f"  → Missing naphies_code: {missing_naphies:,}")
            print(f"  → Low similarity: {low_similarity:,}")
            print(f"  → Missing raw data: {missing_raw:,}")

        print(f"\nTotal Matched: {path1 + path2:,} ({(path1+path2)/total*100:.1f}%)")

        if 'match_score' in self.results.columns:
            path2_scores = self.results[self.results['match_path'] == 'Path2_Matched']['match_score']
            if len(path2_scores) > 0:
                print(f"\nPath 2 Similarity Scores:")
                print(f"  → Average: {path2_scores.mean():.3f}")
                print(f"  → Min: {path2_scores.min():.3f}")
                print(f"  → Max: {path2_scores.max():.3f}")
        print("\n")

    def save_results(self, output_path):
        """Save processed results to Excel file"""
        print(f"Saving results to: {output_path}")

        # Reorder columns to put important ones first
        cols = self.results.columns.tolist()
        priority_cols = ['match_path', 'needs_manual_review', 'match_score', 'naphies_code']
        for col in reversed(priority_cols):
            if col in cols:
                cols.remove(col)
                cols.insert(0, col)

        self.results = self.results[cols]
        self.results.to_excel(output_path, index=False)
        print(f"✓ Results saved successfully!\n")

    def save_no_match(self, output_path):
        """Save no-match rows to a separate Excel file"""
        no_match_data = self.results[self.results['match_path'] == 'No_Match']
        if not no_match_data.empty:
            print(f"Saving {len(no_match_data):,} no-match rows to: {output_path}")
            no_match_data.to_excel(output_path, index=False)
            print(f"✓ No-match rows saved successfully!\n")
        else:
            print("No no-match rows to save.\n")

    def save_review_required(self, output_path):
        """Save all rows that need manual review"""
        review_data = self.results[self.results['needs_manual_review'] == True]
        if not review_data.empty:
            print(f"Saving {len(review_data):,} rows needing manual review to: {output_path}")
            review_data.to_excel(output_path, index=False)
            print(f"✓ Review-required rows saved successfully!\n")
        else:
            print("No rows need manual review.\n")

    def run_pipeline(self, raw_file, pricing_file, naphies_file,
                    output_file='processed_claims.xlsx',
                    no_match_file='no_match_claims.xlsx',
                    review_file='needs_review_claims.xlsx',
                    semantic_threshold=0.75):
        """Run the complete matching pipeline"""
        self.load_data(raw_file, pricing_file, naphies_file)

        initial_rows = len(self.raw_data)
        self.raw_data = self.raw_data.drop_duplicates()
        if len(self.raw_data) < initial_rows:
            print(f"Removed {initial_rows - len(self.raw_data):,} duplicate rows from raw data\n")

        self.filter_non_medicines()
        self.exact_match()
        self.semantic_match(threshold=semantic_threshold)
        self.enrich_with_naphies_data()
        self.generate_summary()
        self.save_results(output_file)
        self.save_no_match(no_match_file)
        self.save_review_required(review_file)

        return self.results
