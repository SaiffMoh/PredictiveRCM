import pandas as pd
import numpy as np
from openai import OpenAI
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import json
from tqdm import tqdm
import warnings
import time
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()  # Load environment variables from .env

class ClaimJustifier:
    def __init__(self, api_key: str = None, batch_size: int = 10, max_workers: int = 5):
        """
        Initialize the claim justifier with OpenAI API
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

        print(f"Initialized ClaimJustifier:")
        print(f"  → Model: {self.model}")
        print(f"  → Batch Size: {batch_size} claims per API call")
        print(f"  → Max Workers: {max_workers} parallel threads\n")

    def load_processed_claims(self, file_path: str):
        """
        Load processed claims from the claim_matcher output
        """
        print("="*60)
        print("LOADING PROCESSED CLAIMS")
        print("="*60)

        self.processed_claims = pd.read_excel(file_path)
        print(f"✓ Loaded {len(self.processed_claims):,} processed claims")

        # Check required columns
        required_cols = ['gender', 'service_description', 'ICD10 Code', 'Diagnoses',
                        'chief_complaint', 'Long Description', 'Definition', 'Includes', 'Guidelines']
        missing_cols = [col for col in required_cols if col not in self.processed_claims.columns]

        if missing_cols:
            print(f"⚠ Warning: Missing columns: {missing_cols}")
            print("→ These claims may have limited context for justification\n")
        else:
            print(f"✓ All required columns present\n")

        # Display distribution by match_path
        if 'match_path' in self.processed_claims.columns:
            print("Claims Distribution by Match Path:")
            print(self.processed_claims['match_path'].value_counts())
            print()

        return self.processed_claims

    def _build_prompt(self, claims_batch: List[Dict]) -> str:
        """
        Build the prompt for batch LLM evaluation
        Acting as an insurance claim reviewer - MUST make a decision (Justified or Not Justified)
        """
        prompt = """You are an insurance claim reviewer for a healthcare insurance company. Your job is to determine if a medical service is justified based on the patient's diagnosis and clinical context.
**CRITICAL: You MUST decide either "Justified" or "Not Justified" for EVERY claim. NO "Flagged for Review" status allowed.**
**SPECIAL RULE - IV INFUSION**:
- If the service_description contains "IV INFUSION" or "INTRAVENOUS INFUSION", you MUST:
  * Status: "Not Justified"
  * Note: "Need medicine for further analysis"
  * needs_manual_review: true
- This is a hard rule - IV infusions always need medicine details before justification
**EVALUATION CRITERIA**:
- If there's ANY doubt about clinical appropriateness → "Not Justified"
- If the diagnosis does NOT clearly support the need for this service → "Not Justified"
- If the service seems excessive, unnecessary, or not clinically indicated → "Not Justified"
- Only mark as "Justified" if there's CLEAR clinical necessity
- If NAPHIES data is missing (Definition, Includes, Guidelines are empty/null), make your best judgment based on:
  * Service description vs. diagnosis alignment
  * Clinical common sense
  * Patient gender appropriateness
  * Chief complaint context
  → If you can't confidently justify it, mark "Not Justified"
**EVALUATION STEPS**:
1. Check for IV INFUSION first (hard rule above)
2. Match the service description against the diagnosis
3. Consider patient gender if relevant (e.g., pregnancy-related services for males = not justified)
4. Check if the service aligns with NAPHIES Guidelines and Includes (if available)
5. Look for red flags: vague diagnosis, unrelated service, excessive treatment
6. **Make a decision: Justified or Not Justified**
**OUTPUT FORMAT** (JSON array):
Return ONLY a JSON array with one object per claim. Each object must have:
{
  "claim_index": <index from input>,
  "status": "Justified" | "Not Justified",
  "note": "<concise reasoning in 1-2 sentences>",
  "needs_manual_review": true | false
}
**IMPORTANT**:
- Be STRICT - when in doubt, mark "Not Justified"
- Keep notes concise but specific (mention key mismatches)
- Set "needs_manual_review" to true for borderline cases, missing data, complex scenarios, or IV infusions
- The "needs_manual_review" flag allows human oversight while you still make the initial decision
---
**CLAIMS TO EVALUATE**:
"""

        for i, claim in enumerate(claims_batch):
            # Extract claim data safely
            gender = self._safe_str(claim.get('gender', 'Unknown'))
            service_desc = self._safe_str(claim.get('service_description', 'N/A'))
            icd10_code = self._safe_str(claim.get('ICD10 Code', 'N/A'))
            diagnoses = self._safe_str(claim.get('Diagnoses', 'N/A'))
            chief_complaint = self._safe_str(claim.get('chief_complaint', 'N/A'))
            long_desc = self._safe_str(claim.get('Long Description', 'N/A'))
            definition = self._safe_str(claim.get('Definition', 'N/A'))
            includes = self._safe_str(claim.get('Includes', 'N/A'))
            guidelines = self._safe_str(claim.get('Guidelines', 'N/A'))
            match_path = self._safe_str(claim.get('match_path', 'Unknown'))

            prompt += f"""
Claim #{i}:
- Index: {claim['_index']}
- Match Path: {match_path} (Path2_Matched = needs extra scrutiny)
- Gender: {gender}
- Service Description (Raw): {service_desc}
- ICD10 Code: {icd10_code}
- Diagnosis: {diagnoses}
- Chief Complaint: {chief_complaint}
- NAPHIES Long Description: {long_desc}
- NAPHIES Definition: {definition}
- NAPHIES Includes: {includes}
- NAPHIES Guidelines: {guidelines}
"""

        prompt += """\n**YOUR RESPONSE** (JSON array only, no other text):"""
        return prompt

    def _safe_str(self, value) -> str:
        """Convert value to string safely, handling NaN/None"""
        if pd.isna(value) or value is None:
            return "N/A"
        return str(value).strip()

    def _check_iv_infusion(self, service_description: str) -> bool:
        """Check if service description contains IV infusion keywords"""
        if pd.isna(service_description):
            return False
        desc_lower = str(service_description).lower()
        iv_keywords = ['iv infusion', 'intravenous infusion', 'i.v. infusion', 'iv therapy']
        return any(keyword in desc_lower for keyword in iv_keywords)

    def _call_llm_batch(self, claims_batch: List[Dict]) -> List[Dict]:
        """
        Call LLM to evaluate a batch of claims
        Returns list of {claim_index, status, note, needs_manual_review}
        """
        prompt = self._build_prompt(claims_batch)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an insurance claim reviewer. Return only valid JSON arrays. You MUST decide Justified or Not Justified for every claim."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000,
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
        Run justification on all claims using multithreading and batching

        Args:
            filter_path: If provided, only process claims with this match_path
                        (e.g., 'Path1_Exact', 'Path2_Matched', 'No_Match')
        """
        if self.processed_claims is None:
            raise ValueError("No processed claims loaded. Call load_processed_claims() first.")

        print("="*60)
        print("RUNNING CLAIM JUSTIFICATION")
        print("="*60)

        # Filter claims if specified
        if filter_path:
            df = self.processed_claims[self.processed_claims['match_path'] == filter_path].copy()
            print(f"Filtering by match_path: {filter_path}")
            print(f"Claims to process: {len(df):,}\n")
        else:
            df = self.processed_claims.copy()
            print(f"Processing all claims: {len(df):,}\n")

        if len(df) == 0:
            print("No claims to process!")
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

            with tqdm(total=len(batches), desc="Processing batches", unit="batch") as pbar:
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

        print(f"\n✓ Processed {len(all_results)} claims")

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

        # POST-PROCESSING: Handle IV infusions
        print("\n" + "="*60)
        print("POST-PROCESSING: IV INFUSION CHECK")
        print("="*60)

        if 'service_description' in self.results.columns:
            iv_mask = self.results['service_description'].apply(self._check_iv_infusion)
            iv_count = iv_mask.sum()

            if iv_count > 0:
                # Override status and note for IV infusions
                self.results.loc[iv_mask, 'status'] = 'Not Justified'
                self.results.loc[iv_mask, 'note'] = 'Need medicine for further analysis'
                self.results.loc[iv_mask, 'needs_manual_review'] = True

                print(f"✓ Identified {iv_count:,} IV infusion claims")
                print(f"→ All marked as 'Not Justified' with note 'Need medicine for further analysis'")
                print(f"→ All flagged for manual review\n")
            else:
                print("→ No IV infusion claims found\n")

        # CONSOLIDATE REVIEW FLAGS: Merge matcher's needs_manual_review with justifier's
        if 'needs_manual_review_x' in self.results.columns and 'needs_manual_review_y' in self.results.columns:
            # Combine both flags (True if either is True)
            self.results['needs_manual_review'] = (
                self.results['needs_manual_review_x'].fillna(False) |
                self.results['needs_manual_review_y'].fillna(False)
            )
            # Drop the duplicate columns
            self.results = self.results.drop(['needs_manual_review_x', 'needs_manual_review_y'], axis=1)
            print("→ Consolidated manual review flags from matcher and justifier\n")

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
        """Display summary of justification results"""
        print("\n" + "="*60)
        print("JUSTIFICATION SUMMARY")
        print("="*60)

        total = len(self.results)
        justified = (self.results['status'] == 'Justified').sum()
        not_justified = (self.results['status'] == 'Not Justified').sum()
        needs_review = self.results['needs_manual_review'].sum()

        print(f"\nTotal Claims Evaluated: {total:,}")

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

        # IV infusion count
        if 'service_description' in self.results.columns:
            iv_count = self.results['service_description'].apply(self._check_iv_infusion).sum()
            if iv_count > 0:
                print(f"\nIV Infusions: {iv_count:,} (all need medicine details)")

        # Sample reasoning
        print("\nSample Justifications:")
        for status in ['Justified', 'Not Justified']:
            sample = self.results[self.results['status'] == status].head(1)
            if not sample.empty:
                note = sample.iloc[0]['note']
                review_flag = "⚠" if sample.iloc[0]['needs_manual_review'] else ""
                print(f"  [{status}] {review_flag}: {note[:100]}...")

        print()

    def save_results(self, output_path: str = 'justified_claims.xlsx'):
        """
        Save results to Excel file
        Includes all original columns plus 'status', 'note', and 'needs_manual_review'
        """
        if self.results is None:
            raise ValueError("No results to save. Run justify_claims() first.")

        print(f"Saving results to: {output_path}")

        # Reorder columns to put status fields at the front
        cols = self.results.columns.tolist()
        status_cols = ['needs_manual_review', 'status', 'note', 'match_path', 'match_score', 'naphies_code']
        for col in status_cols:
            if col in cols:
                cols.remove(col)
        cols = [c for c in status_cols if c in self.results.columns] + cols
        self.results = self.results[cols]

        self.results.to_excel(output_path, index=False)
        print(f"✓ Results saved successfully!")
        print(f"  → Total rows: {len(self.results):,}")
        print(f"  → Columns: {len(self.results.columns)}")
        print(f"  → Justified: {(self.results['status'] == 'Justified').sum():,}")
        print(f"  → Not Justified: {(self.results['status'] == 'Not Justified').sum():,}")
        print(f"  → Needs Manual Review: {self.results['needs_manual_review'].sum():,}\n")

    def run_justification_pipeline(self,
                                  input_file: str,
                                  output_file: str = 'justified_claims.xlsx',
                                  filter_path: str = None):
        """
        Run the complete justification pipeline

        Args:
            input_file: Path to processed claims from claim_matcher
            output_file: Output path for all claims with justification
            filter_path: Optional filter by match_path
        """
        self.load_processed_claims(input_file)
        self.justify_claims(filter_path=filter_path)
        self.save_results(output_file)

        return self.results

# Example usage
if __name__ == "__main__":
    justifier = ClaimJustifier(
        batch_size=10,
        max_workers=5
    )

    results = justifier.run_justification_pipeline(
        input_file='processed_claims.xlsx',
        output_file='justified_claims.xlsx',
        filter_path=None  # Process all claims
    )

    print("Justification pipeline completed successfully!")
    print(f"Results shape: {results.shape}")
