# NBO Analytics Platform

## Overview

The NBO Analytics Platform combines two AI-powered agents - the **ADS Agent** and the **Segmentation Agent** - into a single Streamlit app. Together they take you from a raw SQL database all the way to named, labeled customer segments, without writing a single line of code.

**What the platform does for you:**

- Connects to your database and scans all tables automatically
- Learns what your data means through uploaded documents or a guided chat
- Builds a clean, analysis-ready customer table (the ADS) using AI-generated SQL
- Groups your customers into meaningful segments using machine learning
- Names and describes each segment in plain English
- Lets business users ask questions about segments in a chat
- Exports fully labelled customer data and segment summaries to Excel

All AI processing runs locally on your machine via Ollama. Your data never leaves your environment.

---

## Prerequisites

### Software

| Requirement | Purpose | How to Get It |
|---|---|---|
| Python 3.10+ | Runs the web app | python.org |
| Ollama | Runs the AI model locally | ollama.com |
| qwen2.5:7b | Main AI model | `ollama pull qwen2.5:7b` |
| qwen2.5:3b | Document summarisation | `ollama pull qwen2.5:3b` |

### Python Packages

```
pip install streamlit requests sqlalchemy pandas openpyxl pypdf python-docx scikit-learn matplotlib numpy xlsxwriter
```

For SQL Server / Azure connections only:

```
pip install pyodbc
```

### Hardware

- Minimum: 8 GB RAM
- Recommended: 16 GB RAM
- GPU optional but speeds up AI responses

---

## Pipeline

The platform follows two sequential agents, each with 5 steps.

```
ADS Agent                                    Segmentation Agent
-----------------------------------------    ------------------------------------------
Connect -> Context -> Tables & Goal          Connect -> Table & Goal -> Algorithm
-> Features -> Generate SQL          ----->  -> Role -> Review Segments
```

---

## Installation & Running the App

```bash
# 1. Clone the project
git clone <your-repo-url>
cd nbo-analytics-platform

# 2. Install Python dependencies
pip install streamlit requests sqlalchemy pandas openpyxl pypdf python-docx \
            scikit-learn matplotlib numpy xlsxwriter

# 3. Pull the AI models
ollama pull qwen2.5:7b
ollama pull qwen2.5:3b
```

Start Ollama (run this first, in its own terminal window):

```
ollama serve
```

Launch the app (in a separate terminal):

```
streamlit run combined_app.py
```

Open your browser at **http://localhost:8501**.

---

## The Three Tabs

When the app opens you will see three tabs at the top:

| Tab | Purpose |
|---|---|
| ADS Agent | Build your Analytical Data Set from raw database tables |
| Segmentation Agent | Group customers into segments and explore results |
| Pipeline | Visual, plain-English overview of the complete workflow |

---

## ADS Agent

### Overview

The ADS Agent builds an **Analytical Data Set (ADS)** - a single flat table where each row is one customer and each column is a meaningful piece of information about them (called a feature). Your database likely has dozens of separate tables - transactions, demographics, products, and so on. The ADS Agent combines the most useful information from all of them into one ready-to-use table.

### Pipeline

```
Connect to DB -> Provide Context -> Select Tables & Goal -> Confirm Features -> Generate SQL
```

### Step 1 - Connect to Your Database

Enter your database connection string. The agent connects, scans all tables, and reads their structure (column names, row counts, data types).

Supported formats:

```
SQLite       ->  C:/path/to/file.db
PostgreSQL   ->  postgresql://user:pass@host:5432/dbname
MySQL        ->  mysql+pymysql://user:pass@host:3306/dbname
SQL Server   ->  mssql+pyodbc://user:pass@host/dbname?driver=ODBC+Driver+17+for+SQL+Server
Azure Fabric ->  Data Source=server.fabric.microsoft.com,1433;Initial Catalog=MyDB;Authentication=Active Directory Interactive;Encrypt=True
```

Azure Fabric shortcut: You can paste just the server URL (e.g. `abc123.datawarehouse.fabric.microsoft.com`) and the app will ask for the database name separately.

### Step 2 - Provide Database Context

The AI needs to understand what your data means - not just column names. A column called `cif` or `hassala` means nothing to the AI without context. Providing a description dramatically improves recommendation quality.

**Option A - Upload a Document (Recommended)** Upload a data dictionary, column glossary, or any description of your tables. Supported formats: PDF, Word, Excel, CSV, TXT, Markdown. Long documents are automatically split into chunks, summarised, and merged.

**Option B - Describe via Chat** Answer a few questions from the AI chatbot about your database. After 2+ exchanges a Proceed button appears.

**Option C - Skip** The AI infers meaning from column names only. Works for descriptively named columns but gives weaker recommendations.

**Business Logic Document (Optional)** Upload a separate file describing KPIs, business rules, or model requirements. This is passed to the AI during feature recommendation to align suggestions with your actual business needs.

### Step 3 - Select Tables & Define Your Goal

Select tables - choose the main table(s) your ADS will be built from. The app shows row counts and column counts for every table.

Write your goal - describe what you want to build in plain English. Be specific - this is the single most important input; it drives everything the AI recommends.

Good examples:

> "Identify customers likely to open a term deposit in the next 90 days based on transaction behaviour and demographics"

> "Build a feature set for a credit risk model targeting retail loan applicants"

### Step 4 - Review & Confirm Features

The AI analyses every column in your selected tables - checking data types, null percentages, and distinct value counts - and then produces a feature recommendation.

Feature priority the AI follows:

- **Aggregated/Derived features (preferred)** - computed values like `SUM(transaction_amount)`, `COUNT(products)`, ratios, flags
- **Raw columns** - direct attributes like demographics or status that can't be meaningfully aggregated

For each recommendation the AI explains the source, the SQL formula, and why it helps your goal.

You confirm the final list. Multiselect boxes are pre-filled with the AI's suggestions - add or remove anything freely. You can also pull in columns from other tables and type a JOIN hint if needed.

**Derived Features (Optional)** A chatbot lets you request computed features that don't exist as raw columns:

> "Create a flag for customers whose credit utilisation exceeds 80%"

The AI suggests a SQL formula, validates it against your live database, and shows an Add button to approve it.

Download the feature list as Excel at any time - columns: Type | Table | Feature | Formula | Description.

### Step 5 - Generate & Refine SQL

The AI generates a complete `SELECT` statement (or `CREATE TABLE AS SELECT`) using all your confirmed features, with block-level comments and table aliases throughout.

Refine in plain English - type any change and the AI returns the full updated SQL:

> "Add a WHERE clause to exclude closed accounts"

> "Use INNER JOIN instead of LEFT JOIN for the products table"

> "Wrap this in a CREATE TABLE called ads_credit_risk"

When satisfied, click **Proceed to Segmentation** to hand your ADS directly to the Segmentation Agent.

---

## Segmentation Agent

### Overview

The Segmentation Agent automatically groups your bank's customers into meaningful segments using machine learning - without writing a single line of code. It connects to your database, selects the right columns for your goal, runs a clustering algorithm, names each segment in plain English, and lets you explore and customise the results.

There are two experiences: a technical path for data scientists who want full control, and a business path for non-technical users who want results quickly.

### Pipeline

```
Connect to DB -> Select Table & Goal -> Choose Algorithm -> Choose Role -> Analysis
```

### Step 1 - Connect to Your Database

Enter your database connection string. The agent connects and lists all available tables with their row and column counts. Supported connection formats are the same as the ADS Agent above.

Optionally upload column description documents - PDF, Word, or Excel files that explain what your table's columns mean.

### Step 2 - Select a Table & Define Your Goal

Select a table from the list shown. One table per segmentation run.

Write your segmentation objective - this is required and is the most important input. The AI uses it to decide which columns are relevant and which to ignore.

Good examples:

> "Identify high-value customers for a premium credit card campaign"

> "Group customers by digital channel behaviour to improve app engagement"

> "Segment by financial risk and income to personalise loan offers"

### Step 3 - Choose a Clustering Algorithm

| | K-Means | DBSCAN |
|---|---|---|
| Number of segments | You decide (or AI suggests) | The data decides |
| Outliers | Every customer assigned to a segment | Outlier customers labelled as noise |
| Best for | Known number of groups, easy to explain | Unknown number of groups, irregular shapes |
| Predictability | High | Lower - depends on data density |

Not sure? Start with K-Means. It's more predictable and easier to communicate to stakeholders.

### Step 4 - Choose Your Role

**Data Scientist Path** - full technical control:

- Review schema statistics for every column
- LLM selects features - you can override any selection
- Elbow chart (K-Means) or k-distance graph (DBSCAN) to find optimal parameters
- Manual parameter tuning (K, eps, min_samples)
- Run multiple iterations and compare them side-by-side
- Specific numeric tuning recommendations from the AI

**Business User Path** - automated, jargon-free:

- Everything runs automatically - no interaction needed until results appear
- Segments shown as named cards with plain-English descriptions
- Chat interface to ask questions about any segment
- Business rules to refine segment boundaries
- No statistics or technical terminology

### Step 5A - Data Scientist Analysis

**Feature Selection**

The app shows the full schema with statistics (data type, null %, min/max values) for every column.

- Click **Let LLM select features** - the AI picks 6-30 columns that best match your goal, covering monetary, frequency, recency, demographic, product, and channel dimensions
- Use **Select All**, **Clear All**, or **Reset** to manage the selection manually

**Diagnostic Chart**

- **Elbow Analysis (K-Means)** - tests K=2 through K=10, plots cluster tightness and separation. The elbow is detected automatically and pre-filled as the suggested K.
- **k-Distance Graph (DBSCAN)** - plots how close each customer is to their nearest neighbours. The knee is detected automatically and pre-filled as the suggested eps value.

For tables over 25,000 rows, diagnostics run on a random sample for speed. Results remain representative.

**Results**

| Output | What It Shows |
|---|---|
| Silhouette Score | Cluster quality: >0.5 good, 0.25-0.5 moderate, <0.25 weak |
| Segment Summary Table | AI-generated name, customer count, % of total, plain-English profile |
| Cluster Statistics | Average feature values per segment |
| Feature Thresholds | Min / Mean / Max of each feature per segment in original units |
| Tuning Advice | AI's specific suggestions for improving results |
| Run Comparison | Side-by-side comparison of two consecutive runs with a final recommendation |

### Step 5B - Business User Analysis

Everything runs automatically. Segment cards show: name, customer count, % of total, plain-English profile, and key distinguishing characteristics.

**Business Chat** - ask any plain-English question about the segments:

> "Which segment has the most long-tenured customers?"

> "My company defines high-value as balance above 100,000 - which segment is that?"

> "How do Segment 2 and Segment 3 differ in transaction behaviour?"

The AI answers based only on the actual segment data - it does not invent numbers.

**Customising Segments**

Available via the **Customise Segment Assignments** panel.

- **Create a New Segment** - describe a group in plain English. The AI translates it into a data filter, shows you a preview of matching customers, and moves them into a new named segment on confirmation.

  > "customers with balance above 50,000 and more than 3 products"

- **Rethreshold an Existing Segment** - apply a stricter definition to an existing segment. Customers who don't meet the new threshold are separated into a "remainder" group rather than discarded.

  > Redefine your "High Value" segment from balance > 20,000 to balance > 100,000. The lower-balance members become a new "Former High Value" group.

All overrides show an Undo button and can be removed individually.

**Excel Export**

Click **Prepare Download**, then **Download Excel**.

| Sheet | Contents |
|---|---|
| Labeled Data | Full original table + segment_id and segment_label columns |
| Segment Summary | Name, customer count, % total, profile, key characteristics |
| Cluster Stats | Average feature values per segment |
| Feature Thresholds | Min / Mean / Max per feature per segment |
| Metadata | Run date, table, algorithm, parameters, silhouette score |

Labeled Data is capped at 200,000 rows for Excel compatibility. All summary sheets are always complete.

---

## Pipeline Tab

The Pipeline tab shows you the complete journey from raw database tables to actionable customer segments in a visual, non-technical format. Use it to walk stakeholders through the process or orient yourself before starting a new run.

It also includes a **Quick-Start Guide** - a plain-English summary of every step in both agents on one page.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Ollama is not running" error | Run `ollama serve` in a terminal before launching the app |
| Connection timeout (Azure/Fabric) | Add your IP to the Azure/Fabric firewall allow-list in the portal |
| "Login failed" (SQL Server) | Check credentials; allow browser popups for AAD Interactive auth |
| No tables found after connecting | Verify the connection string points to the correct database |
| Poor AI recommendations | Upload a data dictionary in Step 2 and a business logic document |
| AI feature selection falls back to heuristic | App shows the raw AI response - retry or select columns manually |
| DBSCAN: only 1 cluster found | eps too large - reduce it using the k-distance chart, or switch to K-Means |
| DBSCAN: mostly noise (-1) | eps too small - increase it slightly and re-run |
| Low silhouette score | Try different K (K-Means) or tune eps (DBSCAN); remove highly correlated features |
| "MemoryError" during DBSCAN | Reduce eps or select fewer features |
| Excel export fails | `pip install openpyxl xlsxwriter` |
| PDF parsing unavailable | `pip install pypdf` |
| DOCX parsing unavailable | `pip install python-docx` |
| Segment names look wrong | Upload a column description document in Step 2 and write a clear segmentation goal |
