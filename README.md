# Healthcare Claims Processing System

## 📌 Overview
The **Healthcare Claims Processing System** is a Streamlit-based application designed to automate the matching and justification of healthcare claims using AI and semantic matching. It integrates with NAPHIES reference data, provider pricing lists, and OpenAI’s GPT-4o-mini for claim justification.

---

## 🚀 Features
- **Claim Matching**: Exact and semantic matching of claims to NAPHIES codes.
- **Claim Justification**: AI-powered justification of claims using OpenAI’s GPT-4o-mini.
- **Manual Review Flagging**: Automatically flags claims requiring manual review.
- **Batch Processing**: Supports parallel processing for large datasets.
- **Interactive UI**: User-friendly interface with tabs for setup, processing, results, and downloads.
- **Comprehensive Reporting**: Detailed summaries, match path distribution, and justification status.

---

## 📦 Project Structure

```
.
├── app.py                  # Main Streamlit application
├── claim_matcher.py        # Claim matching logic
├── claim_justifier.py      # Claim justification logic
├── data/
│   ├── providers/           # Provider pricing lists
│   └── naphies/            # NAPHIES reference files
├── output/                 # Processed results
├── .env                    # Environment variables (e.g., OpenAI API key)
└── requirements.txt        # Python dependencies
```

---

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- OpenAI API key (for claim justification)

### Installation
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd healthcare-claims-processing
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Create a `.env` file in the root directory.
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```

4. **Prepare data**:
   - Place provider pricing lists in `data/providers/<provider-name>/`.
   - Place NAPHIES reference files in `data/naphies/`.

---

## 📂 Data Requirements
- **Provider Pricing Lists**: Excel files with columns like `company_name`, `payer_code`, and `naphies_code`.
- **NAPHIES Reference**: Excel file with columns like `SBS Code (Hyphenated)`, `Short Description`, `Long Description`, `Includes`, and `Guidelines`.
- **Raw Claims Data**: Excel file with columns like `company`, `service_code`, `service_description`, `Diagnoses`, and `chief_complaint`.

---

## 🎯 Usage

### 1. Run the Application
```bash
streamlit run app.py
```

### 2. Follow the UI Steps
1. **Setup Tab**:
   - Select a provider.
   - Upload the raw claims data.
   - Verify that the NAPHIES reference is loaded.

2. **Processing Tab**:
   - Configure parameters (semantic threshold, OpenAI API key, batch size, parallel workers).
   - Click **Start Processing** to run the matching and justification pipeline.

3. **Results Tab**:
   - View summary statistics, match path distribution, and justification status.
   - Filter and explore detailed results.

4. **Download Tab**:
   - Download processed files (all results, no-match claims, review-required claims).
   - Reset the system to start a new batch.

---

## 🔧 Configuration
- **Semantic Matching Threshold**: Adjust the threshold for semantic matching (default: `0.75`).
- **OpenAI API Key**: Required for claim justification.
- **Batch Size**: Number of claims processed per API call (default: `10`).
- **Parallel Workers**: Number of parallel threads for processing (default: `5`).

---

## 📊 Outputs
- **All Results**: Complete dataset with matching, enrichment, and justification.
- **No-Match Claims**: Claims that couldn’t be matched to NAPHIES codes.
- **Review Required**: Claims flagged for manual review.

---

## 📝 Notes
- **IV Infusions**: Automatically marked as `Not Justified` and flagged for manual review.
- **Manual Review**: Claims with low similarity scores, missing data, or complex scenarios are flagged for manual review.
- **Error Handling**: The system defaults to `Not Justified` for failed API calls or parsing errors.

---

## 📜 License
This project is licensed under the MIT License.
