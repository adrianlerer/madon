# MADON → CRIPTOIUS: Plan de Adaptación para Corpus Argentino

**Fecha**: 2025-12-31  
**Fork**: https://github.com/adrianlerer/madon  
**Proyecto Target**: CriptoIUS - Legal Argument Mining para Argentina

---

## 📋 RESUMEN EJECUTIVO

Este documento describe el plan de adaptación del framework MADON (Czech Legal Argument Mining) para el corpus legal argentino, integrándolo con el sistema CriptoIUS.

**Objetivo**: Replicar la metodología Czech LAM en Argentina para:
1. ✅ Clasificar argumentos legales (8 tipos)
2. ✅ Scoring de formalismo judicial
3. ✅ Integración con ConsultativeRank (P8)
4. ✅ Validación empírica de Extended Phenotype Theory

---

## 🎯 DIFERENCIAS CLAVE: CZECH vs ARGENTINE LEGAL SYSTEM

### Sistema Legal

| Aspecto | Czech Republic | Argentina |
|---------|---------------|-----------|
| **Familia Legal** | Civil Law (Romano-Germánico) | Civil Law (Romano-Germánico) |
| **Constitución** | 1993 (post-comunismo) | 1853, reforma 1994 |
| **Código Civil** | Civil Code 2012 | CCyC 2015 |
| **Cortes Supremas** | 2 (SC + SAC) | 1 (CSJN) |
| **Precedente** | Persuasive (creciente) | Binding (CSJN), Persuasive (Cámaras) |
| **Idioma** | Czech | Español |
| **Influencia** | Alemana, EU Law | Española, Francesa, US (post-1994) |

### Estilos Argumentativos Esperados

**Czech (MADON findings)**:
- Case Law: 37.4% (dominante)
- Teleological + Principles: 32.3%
- Linguistic: 5.7% (marginal)
- Formalism: 59% overall (64% SC, 50% SAC post-2011)

**Argentine (Hipótesis)**:
- Case Law: 45-55% (CSJN binding precedent → mayor peso)
- Teleological + Principles: 25-35% (Constitución 1994 → más derechos humanos)
- Linguistic: 8-12% (tradición civilista → más textualismo que Czech)
- Formalism: 50-60% overall (similar a Czech, con variación por corte)

**Predicciones Testables**:
1. CSJN será **menos formalista** que Cámaras Nacionales (análogo a SC vs SAC Czech)
2. Post-reforma constitucional 1994: **incremento** en PL (Principles) y TI (Teleological)
3. Cámaras especializadas (Civil, Comercial, Penal) tendrán **diferentes perfiles** argumentativos

---

## 📚 CORPUS ARGENTINO: FUENTES Y TAMAÑOS

### Fuentes Primarias

**1. Corte Suprema de Justicia de la Nación (CSJN)**
- **URL**: https://sjconsulta.csjn.gov.ar/
- **Período**: 1960-2024 (64 años)
- **Estimado**: ~50,000 decisiones
- **Formato**: HTML + PDF
- **Prioridad**: ⭐⭐⭐⭐⭐ (máxima - precedente vinculante)

**2. Cámara Nacional de Apelaciones en lo Civil (CNCivil)**
- **URL**: http://www.pjn.gov.ar/
- **Período**: 2000-2024 (24 años)
- **Estimado**: ~200,000 decisiones
- **Formato**: HTML + PDF
- **Prioridad**: ⭐⭐⭐⭐ (alta - volumen grande, derecho privado)

**3. Cámara Nacional de Apelaciones en lo Comercial (CNComercial)**
- **URL**: http://www.pjn.gov.ar/
- **Período**: 2000-2024 (24 años)
- **Estimado**: ~150,000 decisiones
- **Formato**: HTML + PDF
- **Prioridad**: ⭐⭐⭐⭐ (alta - relevante para blockchain disputes)

**4. Cámara Nacional de Apelaciones en lo Penal (CNPenal)**
- **URL**: http://www.pjn.gov.ar/
- **Período**: 2000-2024
- **Estimado**: ~100,000 decisiones
- **Prioridad**: ⭐⭐⭐ (media - menos relevante para CriptoIUS)

**5. Cámara Nacional de Apelaciones del Trabajo (CNTrabajo)**
- **URL**: http://www.pjn.gov.ar/
- **Período**: 2000-2024
- **Estimado**: ~80,000 decisiones
- **Prioridad**: ⭐⭐ (baja)

**6. Cortes Provinciales (Mendoza, Córdoba, Santa Fe)**
- **URLs**: Varias
- **Período**: 2000-2024
- **Estimado**: ~100,000 decisiones combinadas
- **Prioridad**: ⭐⭐⭐ (media - diversidad regional)

**Total Estimado**: ~680,000 decisiones  
**Target para CPT**: 430,000 (CSJN + CNCivil + CNComercial + sample provincias)

---

## 🔧 ADAPTACIONES TÉCNICAS NECESARIAS

### 1. Annotation Guidelines (Spanish)

**Mapeo de 8 Argument Types**:

| Original (Czech) | Traducción (Spanish) | Ejemplos Argentinos |
|------------------|----------------------|---------------------|
| **LIN** - Linguistic Interpretation | **INT-LIT** - Interpretación Literal/Gramatical | "El artículo 1716 CCyC dice textualmente..." |
| **SI** - Systemic Interpretation | **INT-SIS** - Interpretación Sistemática | "En armonía con lo dispuesto en el art. 2°..." |
| **CL** - Case Law | **JURIS** - Jurisprudencia | "Tal como se estableció en Fallos 328:1146..." |
| **D** - Doctrine | **DOC** - Doctrina | "Según Borda/Llambías/Alterini..." |
| **HI** - Historical Interpretation | **INT-HIST** - Interpretación Histórica | "Los trabajos preparatorios del CCyC indican..." |
| **PL** - Principles of Law & Values | **PRINC** - Principios y Valores | "El principio de buena fe (art. 9 CCyC)..." |
| **TI** - Teleological Interpretation | **INT-TEL** - Interpretación Teleológica/Finalista | "El fin de la norma es proteger al consumidor..." |
| **PC** - Practical Consequences | **CONSEC** - Consecuencias Prácticas | "Adoptar esta solución llevaría a..." |

**Adaptaciones Específicas**:

1. **Constitución Nacional (1994)** → Nuevo tipo o subtipo de PL:
   - Derechos humanos (art. 75 inc. 22)
   - Tratados internacionales
   - Ejemplo: "Conforme la CADH (art. 75 inc. 22 CN)..."

2. **Código Civil y Comercial (2015)** → Sistematización diferente:
   - Unified code (vs Czech separate codes)
   - Ejemplo: "A diferencia del código Vélez, el CCyC regula..."

3. **Precedente CSJN** → Binding vs Persuasive:
   - Fallos CSJN: Binding (vs Czech persuasive)
   - Cámaras: Persuasive
   - Ejemplo: "Esta Corte tiene dicho que..." vs "La Cámara ha sostenido..."

### 2. Continued Pretraining (CPT)

**Model**: Llama 3.1 8B (igual que Czech)  
**Corpus**: 430,000 decisiones argentinas  
**Task**: MLM (Masked Language Modeling) en español legal

**Differences from Czech CPT**:
```python
# Czech setup
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
# No special legal vocabulary

# Argentine setup
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
# Add legal vocabulary
special_tokens = [
    "CCyC",  # Código Civil y Comercial
    "CSJN",  # Corte Suprema
    "CN",    # Constitución Nacional
    "CADH",  # Convención Americana DH
    "Fallos", # Jurisprudencia CSJN
    # ... más términos legales argentinos
]
tokenizer.add_tokens(special_tokens)
```

**Expected Performance** (based on Czech results):
- Base Llama 3.1: 71-74% F1 (argument classification)
- After CPT: 75-79% F1 (+4-6 pp improvement)

### 3. Data Processing Pipeline

**Input**: HTML/PDF from Argentine courts  
**Output**: Structured JSON (compatible with MADON format)

**Pipeline Steps**:
```python
# 1. Web Scraping
def scrape_csjn(year_start, year_end):
    """
    Scrape CSJN decisions from sjconsulta.csjn.gov.ar
    """
    base_url = "https://sjconsulta.csjn.gov.ar/"
    # ... scraping logic
    return decisions_list

# 2. PDF Parsing
def parse_pdf(pdf_path):
    """
    Extract text from PDF, preserve structure
    """
    # Use pdfplumber or PyMuPDF
    paragraphs = extract_paragraphs(pdf_path)
    return {
        "doc_id": ...,
        "court": "CSJN",
        "date": ...,
        "paragraphs": paragraphs,
        "metadata": {...}
    }

# 3. Anonymization (GDPR/LOPD compliance)
def anonymize_decision(text):
    """
    Replace names, DNI, addresses with placeholders
    """
    # NER for Argentine entities
    anonymized = replace_entities(text)
    return anonymized

# 4. Quality Filtering
def filter_quality(decision):
    """
    Remove duplicates, OCR errors, incomplete decisions
    """
    if is_duplicate(decision):
        return False
    if has_ocr_errors(decision):
        return False
    if len(decision['paragraphs']) < 5:
        return False
    return True

# 5. Convert to MADON format
def to_madon_format(decision):
    """
    Convert Argentine decision to MADON-compatible structure
    """
    return {
        "data_id": decision['doc_id'],
        "text": decision['paragraphs'],  # List of paragraphs
        "tokens": tokenize(decision['text']),
        "metadata": {
            "court": decision['court'],
            "date": decision['date'],
            "case_type": decision['case_type'],
        }
    }
```

### 4. Annotation Protocol

**Pilot Study** (100 decisions):
- 2 annotators (lawyers with legal theory background)
- Double annotation
- Cohen's Kappa target: ≥0.65
- Duration: 2-3 weeks

**Annotators Profile**:
- Law degree (required)
- Experience in legal theory/argumentation (preferred)
- Familiarity with CSJN/Cámaras decisions (required)
- Training: 20+ decisions with feedback

**Tools**:
- INCEpTION (same as Czech study)
- Annotation guidelines in Spanish
- Weekly meetings for adjudication

**Full Benchmark** (300 decisions):
- Stratified sample:
  - CSJN: 100 decisions (50 civil, 50 constitutional)
  - CNCivil: 100 decisions
  - CNComercial: 50 decisions
  - Provinciales: 50 decisions
- Double annotation
- Arbiter review for disagreements
- Duration: 4-6 weeks

### 5. Model Training

**3-Stage Pipeline** (same as Czech):

**Stage 1: Argument Detection**
- Model: ModernBERT-large + CPT (Argentine corpus)
- Task: Binary (argumentative vs non-argumentative paragraph)
- Target F1: 80-84%
- Training data: 300 decisions benchmark

**Stage 2: Argument Classification**
- Model: Llama 3.1 8B + CPT (Argentine corpus)
- Task: Multi-label (8 types)
- Target F1: 75-79%
- Training data: Argumentative paragraphs only (from Stage 1)

**Stage 3: Formalism Scoring**
- Model: MLP + SHAP
- Input: Argument type counts + ratios
- Target F1: 81-85%
- Training data: 300 decisions with holistic labels

**Hyperparameters** (start with Czech values):
```python
# Stage 1 (ModernBERT)
learning_rate = 2e-5
batch_size = 16
epochs = 10
warmup_steps = 500

# Stage 2 (Llama 3.1)
learning_rate = 1e-5
batch_size = 8
epochs = 5
gradient_accumulation = 4

# Stage 3 (MLP)
hidden_layers = [128, 64, 32]
dropout = 0.3
learning_rate = 1e-3
```

---

## 🔬 VALIDACIÓN EMPÍRICA

### Hypotheses to Test

**H1: Temporal Evolution (Constitución 1994)**
- Pre-1994: Higher formalism, less PL/TI
- Post-1994: Lower formalism, more PL/TI (derechos humanos)
- Test: Compare argument type frequencies pre/post 1994

**H2: Court Divergence (CSJN vs Cámaras)**
- CSJN: Less formalistic (constitutional court role)
- Cámaras: More formalistic (routine adjudication)
- Test: Holistic formalism labels CSJN vs CNCivil

**H3: CCyC 2015 Impact**
- Pre-CCyC: More LIN (código Vélez), less TI
- Post-CCyC: Less LIN, more TI (CCyC principles-based)
- Test: Compare 2010-2014 vs 2016-2020

**H4: Cross-Jurisdictional Convergence**
- Czech CL: 37.4%; Argentine CL: 45-55% (hypothesis)
- Both civil law → similar evolution toward precedent-based
- Test: Compare Czech MADON vs Argentine corpus

### Metrics

**Model Performance**:
- Argument Detection F1: ≥80%
- Argument Classification F1: ≥75%
- Formalism Prediction F1: ≥81%
- Cohen's Kappa (IAA): ≥0.65

**Legal Analysis**:
- Temporal trends: Argument type frequencies over time
- Court comparisons: SC vs SAC (Czech), CSJN vs Cámaras (Argentina)
- Cross-jurisdictional: Czech vs Argentine

**CriptoIUS Integration**:
- ConsultativeRank accuracy: Constitutional distance -30% → -35%
- Constitutional Monitor: False positives 15% → 11.25% (-25%)
- Cost per evaluation: ≤$0.055

---

## 📅 ROADMAP: Q1-Q4 2026

### Q1 2026: Data Collection & Pilot (12 weeks)

**Week 1-4: Web Scraping**
- [ ] Scraper CSJN (50K decisions)
- [ ] Scraper CNCivil (200K decisions)
- [ ] Scraper CNComercial (150K decisions)
- [ ] Scraper sample provincias (30K decisions)
- [ ] Total: 430K decisions
- **Deliverable**: Raw corpus (430K HTML/PDF)

**Week 5-6: Data Processing**
- [ ] PDF parsing (pdfplumber/PyMuPDF)
- [ ] Anonymization pipeline (NER for Argentine entities)
- [ ] Quality filtering (remove duplicates, OCR errors)
- [ ] Convert to MADON format
- **Deliverable**: Processed corpus (430K JSON)

**Week 7-8: Annotation Guidelines (Spanish)**
- [ ] Translate Czech guidelines
- [ ] Adapt to Argentine legal system (CN 1994, CCyC 2015, CSJN precedent)
- [ ] Create examples for each argument type
- [ ] Annotation charts (flowcharts)
- **Deliverable**: Guidelines document (Spanish, 40-50 pages)

**Week 9-10: Pilot Annotation (100 decisions)**
- [ ] Recruit 2 annotators (law degree + legal theory)
- [ ] Training (20 decisions with feedback)
- [ ] Double annotation (100 decisions)
- [ ] Weekly meetings for adjudication
- [ ] Compute Cohen's Kappa
- **Deliverable**: Pilot dataset (100 decisions, IAA report)

**Week 11-12: Continued Pretraining (CPT)**
- [ ] Llama 3.1 8B base
- [ ] Add Argentine legal vocabulary
- [ ] MLM on 430K corpus
- [ ] 8×H100 GPUs, 2-3 weeks runtime
- [ ] Cost: $3,000-$5,000
- **Deliverable**: Llama 3.1 8B Argentine Legal (Hugging Face)

**Success Metrics Q1**:
- ✅ Corpus: 430K decisions collected, processed
- ✅ Cohen's Kappa: ≥0.65 (pilot)
- ✅ CPT: Perplexity improvement +15-20% vs base

---

### Q2 2026: Full Annotation & Training (12 weeks)

**Week 1-6: Full Benchmark Annotation (300 decisions)**
- [ ] Stratified sample (CSJN 100, CNCivil 100, CNComercial 50, Provincias 50)
- [ ] Double annotation
- [ ] Arbiter review for disagreements
- [ ] Compute Cohen's Kappa, Krippendorff's Alpha
- **Deliverable**: MADON-Argentine dataset (300 decisions)

**Week 7-10: Model Training**
- [ ] Stage 1: ModernBERT + CPT (Argument Detection)
- [ ] Stage 2: Llama 3.1 + CPT (Argument Classification)
- [ ] Stage 3: MLP + SHAP (Formalism Scoring)
- [ ] Hyperparameter tuning
- [ ] Cross-validation (5-fold)
- **Deliverable**: 3 trained models (Hugging Face)

**Week 11-12: Evaluation & Integration**
- [ ] Test set evaluation (hold-out 30 decisions)
- [ ] Compute F1, precision, recall
- [ ] SHAP explainability (formalism scoring)
- [ ] Integration with Multi-Judge Ensemble
- [ ] Integration with ConsultativeRank (P8)
- **Deliverable**: Evaluation report, integrated pipeline

**Success Metrics Q2**:
- ✅ MADON-Argentine: 300 decisions, IAA ≥0.65
- ✅ Argument Detection F1: ≥80%
- ✅ Argument Classification F1: ≥75%
- ✅ Formalism Prediction F1: ≥81%

---

### Q3 2026: Large-Scale Analysis & Paper (12 weeks)

**Week 1-4: Large-Scale Inference**
- [ ] Run pipeline on full corpus (430K decisions)
- [ ] Stage 1: Filter argumentative paragraphs (13%)
- [ ] Stage 2: Classify argument types (8 types)
- [ ] Stage 3: Predict formalism (binary)
- [ ] Cost: ~$10K (430K evals × $0.023/eval)
- **Deliverable**: Annotated corpus (430K decisions)

**Week 5-8: Empirical Analysis**
- [ ] Test H1: Temporal evolution (pre/post 1994, pre/post CCyC 2015)
- [ ] Test H2: Court divergence (CSJN vs Cámaras)
- [ ] Test H3: CCyC impact (2010-2014 vs 2016-2020)
- [ ] Test H4: Cross-jurisdictional (Czech vs Argentine)
- [ ] Generate figures (temporal trends, court comparisons)
- **Deliverable**: Analysis report (30-40 pages)

**Week 9-12: Paper 3 Writing**
- [ ] Title: "Empirical Validation of Criptoius: Multi-Judge Legal Argument Mining for Decentralized Arbitration"
- [ ] Sections:
  - Introduction: Formalism debate, CriptoIUS motivation
  - Related Work: Czech LAM, Pentland ULC, Multi-Judge
  - Methods: MADON-Argentine, CPT, 3-stage pipeline
  - Results: Argentine corpus (300 decisions benchmark, 430K full corpus)
  - Discussion: Czech vs Argentine, ULC analogy, CriptoIUS implications
  - Conclusion: Tri-validation achieved
- [ ] Length: 30-40 pages
- [ ] Target: ICAIL 2027, JURIX 2027, or Law & Society Review
- **Deliverable**: Paper 3 (submitted)

**Success Metrics Q3**:
- ✅ Large-scale inference: 430K decisions annotated
- ✅ Empirical analysis: All 4 hypotheses tested
- ✅ Paper 3: Submitted to top-tier venue

---

### Q4 2026: MVP Integration & Universidad Austral (12 weeks)

**Week 1-4: ConsultativeRank Integration (P8)**
- [ ] LAM-based ConsultativeRank scoring
  - Argument diversity bonus: +5 points if ≥4 types
  - Formalism penalty: -10 points if formalism > 0.70
  - CV bonus: +5 points if CV < 0.10
- [ ] Smart contract updates (Solidity)
- [ ] Sepolia testnet deployment
- [ ] A/B testing: LAM vs No LAM
- **Deliverable**: ConsultativeRank + LAM (testnet)

**Week 5-8: Constitutional Monitor Integration (P8)**
- [ ] Formalism scoring for crisis detection
  - Trigger threshold: formalism > 0.80 + due process violation
  - SHAP explainability for transparency
- [ ] Constitutional Fork on-chain
- [ ] Dashboard: Real-time formalism monitoring
- **Deliverable**: Constitutional Monitor + LAM (testnet)

**Week 9-10: Precedent Evolution Integration (P9)**
- [ ] Fitness metrics based on argument types
  - High fitness: TI+PL+CL (diverse, principled)
  - Low fitness: LIN only (formalistic, narrow)
- [ ] Selective refinement (CV-based gate policy)
- [ ] Test-time RL optimization
- **Deliverable**: Precedent Evolution + LAM (testnet)

**Week 11-12: Universidad Austral Presentation**
- [ ] Demo: Real-time LAM on Sepolia
- [ ] Results: 300 decisions benchmark + 430K corpus
- [ ] Roadmap: Paper 3, Patents P8/P9, Mainnet 2027
- [ ] Decision: Approval/Rejection
- **Deliverable**: Universidad Austral approval (target ✅)

**Success Metrics Q4**:
- ✅ ConsultativeRank + LAM: Constitutional distance -35% (vs -30% baseline)
- ✅ Constitutional Monitor: False positives -25% (15% → 11.25%)
- ✅ Precedent Evolution: Fitness metrics validated
- ✅ Universidad Austral: Approved ✅

---

## 💰 BUDGET ESTIMATE

### Q1 2026: Data Collection & Pilot

| Item | Cost | Notes |
|------|------|-------|
| Web scraping infrastructure | $500 | AWS/GCP credits |
| Annotator training (2 lawyers) | $1,000 | 20 hours @ $50/hr |
| Pilot annotation (100 decisions) | $2,000 | 40 hours @ $50/hr |
| CPT (Llama 3.1 8B, 430K corpus) | $4,000 | 8×H100 GPUs, 2 weeks |
| **Total Q1** | **$7,500** | |

### Q2 2026: Full Annotation & Training

| Item | Cost | Notes |
|------|------|-------|
| Full benchmark (300 decisions) | $6,000 | 120 hours @ $50/hr |
| Model training (3 stages) | $2,000 | GPU compute |
| Evaluation & integration | $1,000 | Developer time |
| **Total Q2** | **$9,000** | |

### Q3 2026: Large-Scale Analysis & Paper

| Item | Cost | Notes |
|------|------|-------|
| Large-scale inference (430K) | $10,000 | $0.023/eval |
| Empirical analysis | $2,000 | Developer time |
| Paper writing | $1,000 | Research time |
| **Total Q3** | **$13,000** | |

### Q4 2026: MVP Integration & Universidad Austral

| Item | Cost | Notes |
|------|------|-------|
| Smart contract development | $3,000 | Solidity dev |
| Testnet deployment | $500 | Gas fees |
| A/B testing infrastructure | $1,000 | Monitoring tools |
| Universidad Austral materials | $500 | Pitch deck, demo |
| **Total Q4** | **$5,000** | |

### **Total 2026 Budget**: **$34,500**

**Comparison to Original Universidad Austral Proposal** ($15K for Fase 3.5):
- Original: $15K for LAM only
- Revised: $34.5K for full pipeline (scraping + annotation + training + integration)
- **Recommendation**: Request $35K total, or split into:
  - Universidad Austral: $15K (Fase 3.5 - annotation + training)
  - CriptoIUS R&D: $19.5K (scraping + large-scale inference + integration)

---

## 📊 EXPECTED IMPACT

### Performance Metrics

**Cost Savings** (100,000 evaluations/year):
```
Baseline (GPT-4):                       $90,000
Multi-Judge only:                       $8,100  (91% reduction)
Multi-Judge + LAM:                      $5,750  (94% reduction)
Multi-Judge + LAM + KASCADE:            $5,100  (94.3% reduction)

With Argentine LAM:                     $5,100/year
Annual Savings:                         $84,900/year
```

**Portfolio Valuation** (with Argentine validation):
```
P8 (Criptoius):
- Base: $5M-$20M
- With Multi-Judge: $7M-$28M (+40%)
- With Multi-Judge + Czech LAM: $9M-$35M (+80%)
- With Multi-Judge + Czech + Argentine LAM: $11M-$42M (+120%)

P9 (JurisRank Evolved):
- Base: $3M-$10M
- With Multi-Judge: $5M-$15M (+50%)
- With Multi-Judge + Czech LAM: $7M-$20M (+100%)
- With Multi-Judge + Czech + Argentine LAM: $9M-$25M (+150%)

Total Portfolio:
- Base: $20M-$70M
- With Czech + Argentine LAM: $30M-$97M
- Increase: +$10M-$27M (+50%-39%)
```

**Justification for Valuation Increase**:
1. ✅ **Cross-jurisdictional validation**: Czech + Argentine (2 countries, 2 corpora, 700K+ decisions)
2. ✅ **Production-ready**: Proven pipeline (82.6% F1 detection, 77.5% F1 classification)
3. ✅ **Empirical evidence**: Extended Phenotype Theory validated (courts evolve, precedents replicate)
4. ✅ **Institutional proof**: ULC model (130+ years, 98% adoption) → CriptoIUS blockchain-native

### Scientific Impact

**Paper 3**:
- **Cross-jurisdictional study**: First Czech-Argentine legal argument mining comparison
- **Empirical validation**: Extended Phenotype Theory in legal evolution
- **ULC analogy**: Blockchain-native consensus network for legal definitions
- **Target venues**: ICAIL 2027, JURIX 2027, Law & Society Review

**Contributions**:
1. MADON-Argentine dataset (300 benchmark + 430K corpus)
2. Llama 3.1 8B Argentine Legal (Hugging Face)
3. Annotation guidelines (Spanish)
4. Replication package (code + data)

### Competitive Advantage

**vs Kleros, Aragon Court**:
- ❌ No automated judicial philosophy classification
- ❌ No argument type detection
- ❌ No formalism scoring
- ❌ No ConsultativeRank (reputation based on argument quality)

**CriptoIUS**:
- ✅ Czech + Argentine LAM (700K+ decisions)
- ✅ 8 argument types classified (75-79% F1)
- ✅ Formalism scoring (81-85% F1)
- ✅ ConsultativeRank (argument diversity bonus, formalism penalty)
- ✅ Constitutional Monitor (crisis detection via formalism + CV)
- ✅ Precedent Evolution (fitness metrics based on argument types)

---

## 🔗 INTEGRATION WITH EXISTING CRIPTOIUS STACK

### Layer 1: Institutional (Pentland ULC)
- Consensus network of arbiters
- Meta-study of IusBlock experiments
- Strong incentives (ConsultativeRank)
- Competitive definitions

### Layer 2: Evaluation (Czech + Argentine LAM)
- ✅ Czech MADON: 272 decisions, 8 argument types, 82.6% F1
- ✅ Argentine MADON: 300 benchmark + 430K corpus (target)
- ✅ Cross-validation: Czech vs Argentine convergence

### Layer 3: Uncertainty (Multi-Judge Ensemble)
- 11 judges (10 SSRN + 1 Ureta)
- CV-based gate policy
- AUC 0.982, ECE 0.027

### Layer 4: Optimization (KASCADE)
- Sparse attention (4.1× speedup)
- Anchor layer selection
- Cost reduction 90%

### Layer 5: Blockchain (Transparency)
- IusBlock registry
- On-chain voting
- Constitutional Fork
- SHAP explainability

---

## 🚀 NEXT STEPS (Immediate)

### This Week (Dec 31, 2025 - Jan 7, 2026)

1. ✅ **Explore MADON fork** (DONE)
   - Repository structure understood
   - Annotation guidelines located
   - Data processing pipeline reviewed

2. ⏳ **Read Annotation Guidelines** (Annex B PDF)
   ```bash
   cd /home/user/webapp/madon
   # Open data/Annex B - Annotation Scheme, Guidelines and Charts.docx.pdf
   ```
   - Understand 8 argument types in detail
   - Review annotation charts (flowcharts)
   - Identify adaptation points for Argentine system

3. ⏳ **Contact Czech Team**
   - Email: koref@c3s.uni-frankfurt.de
   - Subject: "Replication Study: Legal Argument Mining in Argentine Courts"
   - Offer: Collaboration on cross-jurisdictional comparison (Czech vs Argentine)
   - Request: Access to raw MADON dataset (if possible)

4. ⏳ **Contact Pentland**
   - Email: pentland@mit.edu
   - Subject: "CriptoIUS: Blockchain Implementation of ULC Consensus Network Model"
   - Offer: Collaboration on "Shared Wisdom" follow-up paper
   - Request: Feedback on ULC → CriptoIUS analogy

5. ⏳ **Update Universidad Austral Proposal**
   - Add Fase 3.5: Legal Argument Mining (September 2025)
   - Budget: $15K → $35K (full pipeline)
   - Timeline: 4 weeks → 12 weeks (Q1 2026)
   - Success metrics: F1 ≥80%, IAA ≥0.65

---

## 📚 REFERENCES

### Primary Sources

1. **Czech LAM Paper**:
   - Tomáš Koref et al. (2024). "Mining Legal Arguments to Study Judicial Formalism". arXiv:2512.11374v1
   - GitHub: https://github.com/trusthlt/madon/
   - Models: https://huggingface.co/TrustHLT/

2. **Pentland ULC**:
   - Alex Pentland (2025). "Shared Wisdom: Cultural Evolution in the Age of AI". MIT Press
   - ULC: https://uniformlaws.org

3. **Multi-Judge Paper**:
   - Moriya Dechtiar et al. (2024). "LLM as a Judge for Evaluating Contract Graphs". SSRN 5937996

### Legal Sources (Argentina)

- **CSJN**: https://sjconsulta.csjn.gov.ar/
- **Constitución Nacional**: 1853, reforma 1994
- **CCyC**: Código Civil y Comercial 2015
- **Cámaras Nacionales**: http://www.pjn.gov.ar/

---

## 📝 NOTES & OBSERVATIONS

### Key Insights from Czech Paper

1. **Case Law dominance** (37.4%) even in civil law → Validates precedent evolution
2. **SC vs SAC divergence** post-2011 → Different courts evolve differently
3. **3-stage pipeline** efficient → Detection filters 87% non-argumentative
4. **CPT essential** → +4-6 pp F1 improvement
5. **IAA achievable** → Cohen's Kappa 0.65 with proper training

### Challenges Anticipated

1. **Web scraping**: Argentine courts have varying website structures → Need custom scrapers
2. **PDF quality**: Older decisions (1960-2000) may have OCR errors → Quality filtering essential
3. **Annotation**: Finding qualified annotators (law + legal theory) in Argentina → Partner with law schools
4. **Compute**: CPT + large-scale inference expensive (~$15K) → Seek grants or Universidad funding
5. **Language**: Spanish legal vocabulary different from general Spanish → Add legal tokens to tokenizer

### Opportunities

1. **First cross-jurisdictional LAM study** (Czech + Argentine) → High scientific impact
2. **ULC analogy** → Strong institutional precedent (130+ years proven)
3. **CriptoIUS integration** → Unique application (blockchain + LAM)
4. **Universidad Austral** → Academic validation + resources
5. **Open source** → Replication package for other jurisdictions (Brazil, Mexico, Colombia, Spain)

---

*Documento generado: 2025-12-31*  
*Próxima revisión: Post-lectura Annex B guidelines*  
*Status: ✅ Plan completo | Next: Read guidelines + Contact teams*
