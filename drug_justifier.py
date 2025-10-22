import pandas as pd
import numpy as np
from openai import OpenAI
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import json
from tqdm import tqdm
import warnings
import time
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

class DrugJustifier:
    def __init__(self, api_key: str = None, batch_size: int = 10, max_workers: int = 5):
        """
        Initialize the drug claim justifier with OpenAI API
        Uses GPT-4o-mini for cost-effective processing

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            batch_size: Number of claims to process per API call (default: 10)
            max_workers: Number of parallel threads (default: 5)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.processed_claims = None
        self.results = None

        print(f"Initialized DrugJustifier:")
        print(f"  → Model: {self.model}")
        print(f"  → Batch Size: {batch_size} claims per API call")
        print(f"  → Max Workers: {max_workers} parallel threads\n")

    def load_processed_claims(self, file_path: str):
        """
        Load processed drug claims from the drug_matcher output
        """
        print("="*60)
        print("LOADING PROCESSED DRUG CLAIMS")
        print("="*60)

        self.processed_claims = pd.read_excel(file_path)
        print(f"✓ Loaded {len(self.processed_claims):,} processed drug claims")

        # Check required columns
        required_cols = [
            'service_description', 'ICD10 Code', 'Diagnoses', 'chief_complaint',
            'scientific_name', 'chi_icd10_code', 'chi_indication', 'chi_notes'
        ]
        missing_cols = [col for col in required_cols if col not in self.processed_claims.columns]

        if missing_cols:
            print(f"⚠ Warning: Missing columns: {missing_cols}")
            print("→ These claims may have limited context for justification\n")
        else:
            print(f"✓ All required columns present\n")

        # Display distribution by match_path
        if 'match_path' in self.processed_claims.columns:
            print("Drug Claims Distribution by Match Path:")
            print(self.processed_claims['match_path'].value_counts())
            print()

        return self.processed_claims

    def _build_drug_prompt(self, claims_batch: List[Dict]) -> str:
        """
        Build the prompt for batch LLM evaluation - DRUG-SPECIFIC PROMPT
        Acting as a clinical pharmacist/insurance claim reviewer
        """
        prompt = """You are a clinical pharmacist and insurance claim reviewer specializing in medication appropriateness. Your job is to determine if a prescribed medication (drug) is justified based on the patient's diagnosis and clinical context.

**CRITICAL: You MUST decide either "Justified" or "Not Justified" for EVERY drug claim. NO "Flagged for Review" status allowed.**

**DRUG-SPECIFIC EVALUATION CRITERIA**:

1. **ICD-10 Code Matching** (Primary Check):
   - Does the patient's ICD-10 Code from raw data match the CHI ICD-10 Code for this drug?
   - If YES → Strong indication of justification
   - If NO or missing → Proceed to indication analysis

2. **Indication Analysis** (Secondary Check):
   - Does the CHI INDICATION align with the patient's Diagnosis or Chief Complaint?
   - Is the drug approved for treating the patient's condition?
   - Example: If drug indication is "hypertension" and patient diagnosis is "high blood pressure" → Justified
   - Example: If drug indication is "diabetes" but patient has "migraine" → Not Justified

3. **Clinical Appropriateness**:
   - Does the service_description (drug name/dosage) match the scientific name from SFDA/CHI?
   - Are there any contraindications in CHI NOTES that apply to this patient?
   - Does the chief complaint support the need for this medication?
   - Consider patient gender if relevant (e.g., pregnancy-related drugs for males = not justified)

4. **Red Flags for "Not Justified"**:
   - ICD-10 codes don't match AND indication doesn't align with diagnosis
   - Drug is prescribed for a completely unrelated condition
   - CHI NOTES contain contraindications relevant to this patient
   - Missing critical data (no scientific name, no CHI match, no diagnosis)
   - Service description doesn't match the retrieved scientific name

5. **Decision Rule**:
   - **Justified**: Clear match between patient condition and drug indication (ICD-10 match OR indication aligns with diagnosis)
   - **Not Justified**: Mismatch between condition and drug purpose, OR missing critical data, OR contraindications present
   - **When in doubt → "Not Justified"** (err on the side of caution)

**SPECIAL CASES**:
- If CHI data is missing (no chi_icd10_code, chi_indication, or chi_notes): Mark "Not Justified" with note "Insufficient CHI data for justification"
- If scientific name is missing: Mark "Not Justified" with note "Drug not found in SFDA database"
- If multiple indications exist and at least one matches: Can be "Justified"

**OUTPUT FORMAT** (JSON array):
Return ONLY a JSON array with one object per claim. Each object must have:
{
  "claim_index": <index from input>,
  "status": "Justified" | "Not Justified",
  "note": "<concise reasoning in 1-2 sentences>",
  "needs_manual_review": true | false
}

**IMPORTANT**:
- Be STRICT but FAIR - focus on clinical appropriateness
- Keep notes concise but specific (mention key matches/mismatches)
- Set "needs_manual_review" to true for:
  * Borderline cases where indication partially matches
  * Multiple CHI matches found
  * Missing CHI data but service description seems appropriate
  * Complex drug interactions or special populations
- The "needs_manual_review" flag allows human oversight while you still make the initial decision

**EXAMPLES**:

Example 1 - Justified:
- Patient ICD-10: E11.9 (Type 2 Diabetes)
- CHI ICD-10: E11.9
- CHI Indication: "Treatment of type 2 diabetes mellitus"
- Diagnosis: "Type 2 Diabetes Mellitus"
→ Status: "Justified", Note: "ICD-10 codes match exactly, drug indicated for patient's condition"

Example 2 - Not Justified:
- Patient ICD-10: M79.3 (Myalgia)
- CHI ICD-10: I10 (Hypertension)
- CHI Indication: "Management of essential hypertension"
- Diagnosis: "Muscle pain"
→ Status: "Not Justified", Note: "Drug indicated for hypertension but patient has muscle pain - no clinical match"

Example 3 - Not Justified with Review:
- Patient ICD-10: J06.9 (Upper respiratory infection)
- CHI ICD-10: J18.9 (Pneumonia)
- CHI Indication: "Treatment of bacterial respiratory infections"
- Chief Complaint: "Severe cough with fever"
→ Status: "Not Justified", Note: "ICD codes don't match (URI vs Pneumonia), but indication partially relevant", needs_manual_review: true

---

**DRUG CLAIMS TO EVALUATE**:
"""

        for i, claim in enumerate(claims_batch):
            # Extract claim data safely
            service_desc = self._safe_str(claim.get('service_description', 'N/A'))
            icd10_raw = self._safe_str(claim.get('ICD10 Code', 'N/A'))
            diagnoses = self._safe_str(claim.get('Diagnoses', 'N/A'))
            chief_complaint = self._safe_str(claim.get('chief_complaint', 'N/A'))
            scientific_name = self._safe_str(claim.get('scientific_name', 'N/A'))
            chi_icd10 = self._safe_str(claim.get('chi_icd10_code', 'N/A'))
            chi_indication = self._safe_str(claim.get('chi_indication', 'N/A'))
            chi_notes = self._safe_str(claim.get('chi_notes', 'N/A'))
            match_path = self._safe_str(claim.get('match_path', 'Unknown'))
            gender = self._safe_str(claim.get('gender', 'Unknown'))

            prompt += f"""
Drug Claim #{i}:
- Index: {claim['_index']}
- Match Path: {match_path} (CHI_Matched = full data available, SFDA_NoMatch/CHI_NoMatch = incomplete data)
- Patient Gender: {gender}

**RAW CLAIM DATA**:
- Service Description (Prescribed Drug): {service_desc}
- Patient ICD-10 Code: {icd10_raw}
- Patient Diagnosis: {diagnoses}
- Chief Complaint: {chief_complaint}

**SFDA/CHI REFERENCE DATA**:
- Scientific Name (from SFDA): {scientific_name}
- CHI ICD-10 Code: {chi_icd10}
- CHI Indication: {chi_indication}
- CHI Notes/Contraindications: {chi_notes}

**YOUR TASK**: Compare patient's condition (ICD-10, Diagnosis, Chief Complaint) with drug's approved use (CHI ICD-10, Indication, Notes). Is this drug appropriate for this patient?
"""

        prompt += """\n**YOUR RESPONSE** (JSON array only, no other text):"""
        return prompt

    def _safe_str(self, value) -> str:
        """Convert value to string safely, handling NaN/None"""
        if pd.isna(value) or value is None:
            return "N/A"
        return str(value).strip()

    def _call_llm_batch(self, claims_batch: List[Dict]) -> List[Dict]:
        """
        Call LLM to evaluate a batch of drug claims
        Returns list of {claim_index, status, note, needs_manual_review}
        """
        prompt = self._build_drug_prompt(claims_batch)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a clinical pharmacist and insurance claim reviewer. Return only valid JSON arrays. You MUST decide Justified or Not Justified for every drug claim based on clinical appropriateness."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2500,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content.strip()

                # Parse JSON response
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        for key in ['results', 'claims', 'evaluations', 'data']:
                            if key in parsed and isinstance(parsed[key], list):
                                parsed = parsed[key]
                                break
                        if isinstance(parsed, dict):
                            for value in parsed.values():
                                if isinstance(value, list):
                                    parsed = value
                                    break

                    if not isinstance(parsed, list):
                        raise ValueError(f"Expected list, got {type(parsed)}")

                    # Validate and normalize structure
                    for item in parsed:
                        if not all(k in item for k in ['claim_index', 'status', 'note']):
                            raise ValueError(f"Missing required keys in: {item}")
                        if item['status'] not in ['Justified', 'Not Justified']:
                            # Force invalid statuses to "Not Justified"
                            item['status'] = 'Not Justified'
                            item['needs_manual_review'] = True
                        # Ensure needs_manual_review exists
                        if 'needs_manual_review' not in item:
                            item['needs_manual_review'] = False

                    return parsed

                except json.JSONDecodeError as e:
                    print(f"⚠ JSON decode error (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        return [
                            {
                                'claim_index': claim['_index'],
                                'status': 'Not Justified',
                                'note': f'LLM response parsing failed - defaulting to Not Justified',
                                'needs_manual_review': True
                            }
                            for claim in claims_batch
                        ]
                    time.sleep(2)
                    continue

            except Exception as e:
                print(f"⚠ API error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return [
                        {
                            'claim_index': claim['_index'],
                            'status': 'Not Justified',
                            'note': f'API call failed - defaulting to Not Justified',
                            'needs_manual_review': True
                        }
                        for claim in claims_batch
                    ]
                time.sleep(2)

        return [
            {
                'claim_index': claim['_index'],
                'status': 'Not Justified',
                'note': 'Unexpected error - defaulting to Not Justified',
                'needs_manual_review': True
            }
            for claim in claims_batch
        ]

    def _process_batch_worker(self, batch: List[Dict]) -> List[Dict]:
        """
        Worker function for parallel processing
        """
        return self._call_llm_batch(batch)

    def justify_claims(self, filter_path: str = None):
        """
        Run justification on all drug claims using multithreading and batching

        Args:
            filter_path: If provided, only process claims with this match_path
                        (e.g., 'CHI_Matched', 'CHI_NoMatch', 'SFDA_NoMatch')
        """
        if self.processed_claims is None:
            raise ValueError("No processed claims loaded. Call load_processed_claims() first.")

        print("="*60)
        print("RUNNING DRUG CLAIM JUSTIFICATION")
        print("="*60)

        # Filter claims if specified
        if filter_path:
            df = self.processed_claims[self.processed_claims['match_path'] == filter_path].copy()
            print(f"Filtering by match_path: {filter_path}")
            print(f"Claims to process: {len(df):,}\n")
        else:
            df = self.processed_claims.copy()
            print(f"Processing all drug claims: {len(df):,}\n")

        if len(df) == 0:
            print("No drug claims to process!")
            return None

        # Add index for tracking
        df['_index'] = df.index

        # Convert to list of dicts for batch processing
        claims_list = df.to_dict('records')

        # Create batches
        batches = [claims_list[i:i+self.batch_size] for i in range(0, len(claims_list), self.batch_size)]
        print(f"Created {len(batches)} batches of ~{self.batch_size} claims each")
        print(f"Parallel workers: {self.max_workers}\n")

        # Process batches in parallel
        all_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {executor.submit(self._process_batch_worker, batch): i
                              for i, batch in enumerate(batches)}

            with tqdm(total=len(batches), desc="Processing drug claim batches", unit="batch") as pbar:
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)
                        pbar.update(1)
                    except Exception as e:
                        print(f"\n⚠ Batch {batch_idx} failed: {e}")
                        failed_batch = batches[batch_idx]
                        for claim in failed_batch:
                            all_results.append({
                                'claim_index': claim['_index'],
                                'status': 'Not Justified',
                                'note': f'Batch processing failed - defaulting to Not Justified',
                                'needs_manual_review': True
                            })
                        pbar.update(1)

        print(f"\n✓ Processed {len(all_results)} drug claims")

        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)

        # Ensure all required columns exist in results_df
        for col in ['status', 'note', 'needs_manual_review']:
            if col not in results_df.columns:
                if col == 'status':
                    results_df[col] = 'Not Justified'
                elif col == 'note':
                    results_df[col] = 'Missing data'
                elif col == 'needs_manual_review':
                    results_df[col] = True

        # Merge back with original data
        self.results = df.merge(
            results_df,
            left_on='_index',
            right_on='claim_index',
            how='left'
        )

        # Clean up temporary columns
        self.results = self.results.drop(['_index', 'claim_index'], axis=1)

        # CONSOLIDATE REVIEW FLAGS: Merge matcher's needs_manual_review with justifier's
        if 'needs_manual_review_x' in self.results.columns and 'needs_manual_review_y' in self.results.columns:
            # Combine both flags (True if either is True)
            self.results['needs_manual_review'] = (
                self.results['needs_manual_review_x'].fillna(False) |
                self.results['needs_manual_review_y'].fillna(False)
            )
            # Drop the duplicate columns
            self.results = self.results.drop(['needs_manual_review_x', 'needs_manual_review_y'], axis=1)
            print("→ Consolidated manual review flags from drug matcher and justifier\n")

        # Fill any missing results (safety)
        if 'status' in self.results.columns:
            self.results['status'] = self.results['status'].fillna('Not Justified')
        else:
            self.results['status'] = 'Not Justified'

        if 'note' in self.results.columns:
            self.results['note'] = self.results['note'].fillna('Processing failed - defaulting to Not Justified')
        else:
            self.results['note'] = 'Processing failed - defaulting to Not Justified'

        if 'needs_manual_review' not in self.results.columns:
            self.results['needs_manual_review'] = True
        else:
            self.results['needs_manual_review'] = self.results['needs_manual_review'].fillna(True)

        self._display_summary()

        return self.results

    def _display_summary(self):
        """Display summary of drug justification results"""
        print("\n" + "="*60)
        print("DRUG JUSTIFICATION SUMMARY")
        print("="*60)

        total = len(self.results)
        justified = (self.results['status'] == 'Justified').sum()
        not_justified = (self.results['status'] == 'Not Justified').sum()
        needs_review = self.results['needs_manual_review'].sum()

        print(f"\nTotal Drug Claims Evaluated: {total:,}")

        print(f"\n✓ Justified: {justified:,} ({justified/total*100:.1f}%)")
        print(f"✗ Not Justified: {not_justified:,} ({not_justified/total*100:.1f}%)")
        print(f"⚠ Needs Manual Review: {needs_review:,} ({needs_review/total*100:.1f}%)")

        # Breakdown by match_path
        if 'match_path' in self.results.columns:
            print("\nBreakdown by Match Path:")
            for path in self.results['match_path'].unique():
                path_df = self.results[self.results['match_path'] == path]
                path_justified = (path_df['status'] == 'Justified').sum()
                path_total = len(path_df)
                print(f"  {path}: {path_justified}/{path_total} justified ({path_justified/path_total*100:.1f}%)")

        # Sample reasoning
        print("\nSample Drug Justifications:")
        for status in ['Justified', 'Not Justified']:
            sample = self.results[self.results['status'] == status].head(1)
            if not sample.empty:
                note = sample.iloc[0]['note']
                review_flag = "⚠ " if sample.iloc[0]['needs_manual_review'] else ""
                sci_name = sample.iloc[0].get('scientific_name', 'N/A')
                print(f"  [{status}] {review_flag}{sci_name[:40]}: {note[:80]}...")

        print()

    def save_results(self, output_path: str = 'drug_justified_claims.xlsx'):
        """
        Save results to Excel file
        Includes all original columns plus 'status', 'note', and 'needs_manual_review'
        """
        if self.results is None:
            raise ValueError("No results to save. Run justify_claims() first.")

        print(f"Saving drug justification results to: {output_path}")

        # Reorder columns to put status fields at the front
        cols = self.results.columns.tolist()
        status_cols = [
            'needs_manual_review', 'status', 'note', 'match_path', 
            'scientific_name', 'chi_icd10_code', 'chi_indication', 'chi_notes'
        ]
        for col in status_cols:
            if col in cols:
                cols.remove(col)
        cols = [c for c in status_cols if c in self.results.columns] + cols
        self.results = self.results[cols]

        self.results.to_excel(output_path, index=False)
        print(f"✓ Drug justification results saved successfully!")
        print(f"  → Total rows: {len(self.results):,}")
        print(f"  → Columns: {len(self.results.columns)}")
        print(f"  → Justified: {(self.results['status'] == 'Justified').sum():,}")
        print(f"  → Not Justified: {(self.results['status'] == 'Not Justified').sum():,}")
        print(f"  → Needs Manual Review: {self.results['needs_manual_review'].sum():,}\n")

    def run_justification_pipeline(self,
                                  input_file: str,
                                  output_file: str = 'drug_justified_claims.xlsx',
                                  filter_path: str = None):
        """
        Run the complete drug justification pipeline

        Args:
            input_file: Path to processed drug claims from drug_matcher
            output_file: Output path for all drug claims with justification
            filter_path: Optional filter by match_path
        """
        self.load_processed_claims(input_file)
        self.justify_claims(filter_path=filter_path)
        self.save_results(output_file)

        return self.results


# Example usage
if __name__ == "__main__":
    justifier = DrugJustifier(
        batch_size=10,
        max_workers=5
    )

    results = justifier.run_justification_pipeline(
        input_file='output/drug_matched_claims.xlsx',
        output_file='output/drug_justified_claims.xlsx',
        filter_path=None  # Process all drug claims
    )

    print("Drug justification pipeline completed successfully!")
    print(f"Results shape: {results.shape}")