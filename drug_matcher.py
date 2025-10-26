import pandas as pd
import numpy as np
import warnings
import re

warnings.filterwarnings('ignore')

class DrugMatcher:
    def __init__(self):
        """Initialize the drug matcher"""
        self.raw_data = None
        self.sfda_ref = None
        self.chi_ref = None
        self.results = None
        print("Initialized DrugMatcher for medicine claims processing\n")

    def load_data(self, raw_file_path, sfda_file_path, chi_file_path, sheet_name='Worksheet'):
        """
        Load all required data files for drug matching
        """
        print("="*60)
        print("LOADING DRUG DATA FILES")
        print("="*60)

        # Load raw data with specified sheet name
        try:
            self.raw_data = pd.read_excel(raw_file_path, sheet_name=sheet_name)
            print(f"✓ Loaded {len(self.raw_data):,} rows from raw data (sheet: '{sheet_name}')")
        except ValueError as e:
            # Try to read the first sheet if specified sheet not found
            print(f"⚠ Sheet '{sheet_name}' not found, reading first sheet...")
            self.raw_data = pd.read_excel(raw_file_path, sheet_name=0)
            print(f"✓ Loaded {len(self.raw_data):,} rows from raw data (first sheet)")

        # Remove 'membership' column if it exists
        if 'membership' in self.raw_data.columns:
            self.raw_data = self.raw_data.drop(columns=['membership'])

        # Load SFDA reference
        self.sfda_ref = pd.read_excel(sfda_file_path)
        print(f"✓ Loaded {len(self.sfda_ref):,} rows from SFDA reference")

        # Load CHI reference
        self.chi_ref = pd.read_excel(chi_file_path)
        print(f"✓ Loaded {len(self.chi_ref):,} rows from CHI reference\n")

    def _clean_service_code(self, code):
        """
        Clean service_code for matching with SFDA RegisterNumber
        """
        if pd.isna(code):
            return None
        code_str = str(code).strip().upper()
        # Remove 'TAW-' prefix if present (case-insensitive)
        if code_str.upper().startswith('TAW-'):
            code_str = code_str[4:]
        # Remove common separators and special chars
        code_str = re.sub(r'[^A-Z0-9]', '', code_str)
        return code_str if code_str else None

    def _clean_scientific_name(self, name):
        """
        Clean scientific name for matching
        """
        if pd.isna(name):
            return None
        name_str = str(name).strip().lower()
        # Remove extra spaces
        name_str = re.sub(r'\s+', ' ', name_str)
        return name_str if name_str else None

    def filter_medicines(self):
        """
        Filter raw data to only include medicine claims
        """
        print("="*60)
        print("FILTERING MEDICINE CLAIMS")
        print("="*60)

        initial_count = len(self.raw_data)

        if 'service_type' in self.raw_data.columns:
            medicine_keywords = ['medicine', 'drug', 'pharmacy', 'medication']
            mask = self.raw_data['service_type'].str.lower().str.contains(
                '|'.join(medicine_keywords),
                na=False
            )
            self.raw_data = self.raw_data[mask].copy()

        filtered_count = len(self.raw_data)
        print(f"Initial rows: {initial_count:,}")
        print(f"Medicine claims: {filtered_count:,}")
        print(f"Removed: {initial_count - filtered_count:,} non-medicine claims\n")

    def match_sfda(self):
        """
        Step 1: Match service_code with SFDA RegisterNumber
        """
        print("="*60)
        print("STEP 1: MATCHING WITH SFDA (RegisterNumber → Scientific Name)")
        print("="*60)

        # Initialize columns
        self.raw_data['needs_manual_review'] = False
        self.raw_data['match_path'] = 'Pending'
        self.raw_data['mismatch_reason'] = None

        # Clean service_code in raw data
        print("Cleaning service codes...")
        self.raw_data['service_code_cleaned'] = self.raw_data['service_code'].apply(
            self._clean_service_code
        )

        # Clean RegisterNumber in SFDA
        self.sfda_ref['RegisterNumber_cleaned'] = self.sfda_ref['RegisterNumber'].apply(
            self._clean_service_code
        )

        # Prepare SFDA lookup dictionary
        sfda_lookup = self.sfda_ref.set_index('RegisterNumber_cleaned')[
            ['RegisterNumber', 'Scientific Name']
        ].to_dict('index')

        print(f"SFDA lookup table size: {len(sfda_lookup):,} entries\n")

        # Match service codes
        matched_count = 0
        unmatched_count = 0

        scientific_names = []
        sfda_register_numbers = []
        match_paths = []
        mismatch_reasons = []

        for idx, row in self.raw_data.iterrows():
            cleaned_code = row['service_code_cleaned']

            if pd.isna(cleaned_code) or cleaned_code not in sfda_lookup:
                scientific_names.append(None)
                sfda_register_numbers.append(None)
                match_paths.append('SFDA_NoMatch')
                mismatch_reasons.append(
                    'Service code not found in SFDA' if cleaned_code
                    else 'Missing service_code in raw data'
                )
                unmatched_count += 1
            else:
                sfda_data = sfda_lookup[cleaned_code]
                scientific_names.append(sfda_data.get('Scientific Name'))
                sfda_register_numbers.append(sfda_data.get('RegisterNumber'))
                match_paths.append('SFDA_Matched')
                mismatch_reasons.append(None)
                matched_count += 1

        # Add results to dataframe
        self.raw_data['scientific_name'] = scientific_names
        self.raw_data['sfda_register_number'] = sfda_register_numbers
        self.raw_data['match_path'] = match_paths
        self.raw_data['mismatch_reason'] = mismatch_reasons

        # Flag unmatched for review
        self.raw_data.loc[
            self.raw_data['match_path'] == 'SFDA_NoMatch',
            'needs_manual_review'
        ] = True

        match_pct = (matched_count / len(self.raw_data) * 100) if len(self.raw_data) > 0 else 0

        print(f"✓ SFDA matches: {matched_count:,} / {len(self.raw_data):,} ({match_pct:.1f}%)")
        print(f"✗ SFDA no match: {unmatched_count:,} ({(unmatched_count/len(self.raw_data)*100):.1f}%)")

        if matched_count > 0:
            sample_names = self.raw_data[
                self.raw_data['match_path'] == 'SFDA_Matched'
            ]['scientific_name'].head(3).tolist()
            print(f"\nSample scientific names retrieved: {sample_names}\n")

        self.results = self.raw_data
        return self.results

    def match_chi(self):
        """
        Step 2: Match Scientific Name with CHI
        """
        print("="*60)
        print("STEP 2: MATCHING WITH CHI (Scientific Name → ICD10/INDICATION/NOTES)")
        print("="*60)

        # Only process rows that matched SFDA
        sfda_matched = self.raw_data['match_path'] == 'SFDA_Matched'

        if sfda_matched.sum() == 0:
            print("No SFDA matches to process with CHI.\n")
            return self.raw_data

        print(f"Processing {sfda_matched.sum():,} SFDA-matched rows with CHI...")

        # Clean scientific names in CHI
        self.chi_ref['scientific_name_cleaned'] = self.chi_ref['SCIENTIFIC NAME'].apply(
            self._clean_scientific_name
        )

        # Prepare CHI lookup dictionary
        chi_lookup = {}
        for idx, row in self.chi_ref.iterrows():
            sci_name = row['scientific_name_cleaned']
            if pd.notna(sci_name):
                if sci_name not in chi_lookup:
                    chi_lookup[sci_name] = []
                chi_lookup[sci_name].append({
                    'ICD10_CODE': row.get('ICD 10 CODE'),
                    'INDICATION': row.get('INDICATION'),
                    'NOTES': row.get('NOTES'),
                    'SCIENTIFIC_NAME': row.get('SCIENTIFIC NAME')
                })

        print(f"CHI lookup table size: {len(chi_lookup):,} unique scientific names\n")

        # Match with CHI
        chi_matched_count = 0
        chi_unmatched_count = 0

        chi_icd10_codes = []
        chi_indications = []
        chi_notes = []
        chi_scientific_names = []

        for idx, row in self.raw_data.iterrows():
            if row['match_path'] != 'SFDA_Matched':
                # Keep existing values for non-SFDA matches
                chi_icd10_codes.append(None)
                chi_indications.append(None)
                chi_notes.append(None)
                chi_scientific_names.append(None)
                continue

            sci_name_cleaned = self._clean_scientific_name(row['scientific_name'])

            if pd.isna(sci_name_cleaned) or sci_name_cleaned not in chi_lookup:
                chi_icd10_codes.append(None)
                chi_indications.append(None)
                chi_notes.append(None)
                chi_scientific_names.append(None)
                self.raw_data.at[idx, 'match_path'] = 'CHI_NoMatch'
                self.raw_data.at[idx, 'mismatch_reason'] = 'Scientific name not found in CHI'
                self.raw_data.at[idx, 'needs_manual_review'] = True
                chi_unmatched_count += 1
            else:
                # Get first match (can be enhanced to handle multiple matches)
                chi_data = chi_lookup[sci_name_cleaned][0]
                chi_icd10_codes.append(chi_data['ICD10_CODE'])
                chi_indications.append(chi_data['INDICATION'])
                chi_notes.append(chi_data['NOTES'])
                chi_scientific_names.append(chi_data['SCIENTIFIC_NAME'])
                self.raw_data.at[idx, 'match_path'] = 'CHI_Matched'
                self.raw_data.at[idx, 'mismatch_reason'] = None
                chi_matched_count += 1

                # Flag for review if multiple matches exist
                if len(chi_lookup[sci_name_cleaned]) > 1:
                    self.raw_data.at[idx, 'needs_manual_review'] = True
                    self.raw_data.at[idx, 'mismatch_reason'] = f'Multiple CHI matches found ({len(chi_lookup[sci_name_cleaned])})'

        # Add CHI data to results
        self.raw_data['chi_icd10_code'] = chi_icd10_codes
        self.raw_data['chi_indication'] = chi_indications
        self.raw_data['chi_notes'] = chi_notes
        self.raw_data['chi_scientific_name'] = chi_scientific_names

        total_sfda_matched = sfda_matched.sum()
        chi_match_pct = (chi_matched_count / total_sfda_matched * 100) if total_sfda_matched > 0 else 0

        print(f"✓ CHI matches: {chi_matched_count:,} / {total_sfda_matched:,} ({chi_match_pct:.1f}%)")
        print(f"✗ CHI no match: {chi_unmatched_count:,}\n")

        if chi_matched_count > 0:
            sample_indications = self.raw_data[
                self.raw_data['match_path'] == 'CHI_Matched'
            ]['chi_indication'].dropna().head(3).tolist()
            print(f"Sample CHI indications: {[str(ind)[:60] + '...' for ind in sample_indications]}\n")

        self.results = self.raw_data
        return self.results

    def generate_summary(self):
        """Generate and display summary statistics"""
        print("="*60)
        print("DRUG MATCHING SUMMARY")
        print("="*60)

        total = len(self.results)
        sfda_matched = (self.results['match_path'].str.contains('SFDA_Matched|CHI', na=False)).sum()
        chi_matched = (self.results['match_path'] == 'CHI_Matched').sum()
        sfda_no_match = (self.results['match_path'] == 'SFDA_NoMatch').sum()
        chi_no_match = (self.results['match_path'] == 'CHI_NoMatch').sum()
        needs_review = self.results['needs_manual_review'].sum()

        print(f"\nTotal drug claims processed: {total:,}")
        print(f"\n✓ SFDA matched: {sfda_matched:,} ({sfda_matched/total*100:.1f}%)")
        print(f"✓ CHI matched (full pipeline): {chi_matched:,} ({chi_matched/total*100:.1f}%)")
        print(f"\n✗ SFDA no match: {sfda_no_match:,} ({sfda_no_match/total*100:.1f}%)")
        print(f"✗ CHI no match: {chi_no_match:,} ({chi_no_match/total*100:.1f}%)")
        print(f"\n⚠ Needs manual review: {needs_review:,} ({needs_review/total*100:.1f}%)")

        # Breakdown of review reasons
        if needs_review > 0:
            print("\nManual Review Breakdown:")
            review_rows = self.results[self.results['needs_manual_review'] == True]
            for reason in review_rows['mismatch_reason'].unique():
                if pd.notna(reason):
                    count = (review_rows['mismatch_reason'] == reason).sum()
                    print(f"  → {reason}: {count:,}")

        print()

    def save_results(self, output_path):
        """Save drug matching results to Excel file"""
        print(f"Saving drug matching results to: {output_path}")

        # Reorder columns to put important ones first
        cols = self.results.columns.tolist()
        priority_cols = [
            'match_path', 'needs_manual_review', 'scientific_name',
            'chi_icd10_code', 'chi_indication', 'chi_notes'
        ]
        for col in reversed(priority_cols):
            if col in cols:
                cols.remove(col)
                cols.insert(0, col)

        self.results = self.results[cols]
        self.results.to_excel(output_path, index=False)
        print(f"✓ Drug matching results saved successfully!\n")

    def save_no_match(self, output_path):
        """Save no-match drug claims to a separate Excel file"""
        no_match_data = self.results[
            (self.results['match_path'] == 'SFDA_NoMatch') |
            (self.results['match_path'] == 'CHI_NoMatch')
        ]
        if not no_match_data.empty:
            print(f"Saving {len(no_match_data):,} no-match drug claims to: {output_path}")
            no_match_data.to_excel(output_path, index=False)
            print(f"✓ No-match drug claims saved successfully!\n")
        else:
            print("No no-match drug claims to save.\n")

    def save_review_required(self, output_path):
        """Save all drug claims that need manual review"""
        review_data = self.results[self.results['needs_manual_review'] == True]
        if not review_data.empty:
            print(f"Saving {len(review_data):,} drug claims needing manual review to: {output_path}")
            review_data.to_excel(output_path, index=False)
            print(f"✓ Drug review-required claims saved successfully!\n")
        else:
            print("No drug claims need manual review.\n")

    def run_pipeline(self, raw_file, sfda_file, chi_file,
                    output_file='drug_matched_claims.xlsx',
                    no_match_file='drug_no_match_claims.xlsx',
                    review_file='drug_needs_review_claims.xlsx',
                    sheet_name='Worksheet'):
        """
        Run the complete drug matching pipeline
        """
        self.load_data(raw_file, sfda_file, chi_file, sheet_name=sheet_name)

        # Remove duplicates
        initial_rows = len(self.raw_data)
        self.raw_data = self.raw_data.drop_duplicates()
        if len(self.raw_data) < initial_rows:
            print(f"Removed {initial_rows - len(self.raw_data):,} duplicate rows from raw data\n")

        self.filter_medicines()
        self.match_sfda()
        self.match_chi()
        self.generate_summary()
        self.save_results(output_file)
        self.save_no_match(no_match_file)
        self.save_review_required(review_file)

        return self.results
