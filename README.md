# DoctTune
# DocTune

**Turn your documents into a fine-tuned language model, completely free.**

DocTune is a simple, end-to-end pipeline that reads documents in different formats (PDF, DOCX, TXT, CSV), converts them into a training dataset, and fine-tunes a small open-source LLM using **LoRA / QLoRA** with **Unsloth**. The whole thing runs on a **free Google Colab T4 GPU** and finishes in a few minutes of training.

It is built as an exploration project: small, readable, and easy to run top to bottom.

---

## Table of Contents

1. [What this project does](#what-this-project-does)
2. [Pipeline overview](#pipeline-overview)
3. [Tech stack](#tech-stack)
4. [Step-by-step: what happens in each stage](#step-by-step-what-happens-in-each-stage)
5. [Results](#results)
6. [How to run it](#how-to-run-it)
7. [Project structure](#project-structure)
8. [Settings explained](#settings-explained)
9. [Limitations](#limitations)
10. [Future improvements](#future-improvements)

---

## What this project does

Fine-tuning a model on your own documents normally needs a powerful GPU and a lot of setup. DocTune shows that you can do it for free:

- **Input:** a folder of documents (`.pdf`, `.docx`, `.txt`, `.csv`)
- **Output:** a LoRA adapter (a small file) that teaches a base model the language and ideas of your documents

In my test run, I used a PDF about AI system design interviews. After fine-tuning, the model answered in the style and vocabulary of that book (layers, evals, guardrails, latency and cost), while the base model gave generic answers.

---

## Pipeline overview

```
 Documents            Parse & clean         Chunk text          HF Dataset
 (PDF/DOCX/     -->   extract raw text -->  ~1000 chars  -->   {"text", "source"}
  TXT/CSV)                                   per chunk
                                                                     |
                                                                     v
 Saved LoRA     <--   Fine-tune with   <--  Load 4-bit    <--  Add EOS token
 adapter              SFTTrainer            Llama-3.2-1B       to every chunk
 (Google Drive)       (200 steps)           + LoRA adapters
```

---

## Tech stack

| Tool | Role |
| --- | --- |
| **Google Colab (free T4, ~15 GB)** | Free GPU environment |
| **Unsloth** | Fast, memory-efficient LoRA/QLoRA fine-tuning |
| **Llama-3.2-1B-Instruct (4-bit)** | Base model (`unsloth/Llama-3.2-1B-Instruct-bnb-4bit`) |
| **TRL (SFTTrainer / SFTConfig)** | Supervised fine-tuning loop |
| **Hugging Face Datasets** | Dataset format |
| **pypdf, python-docx, pandas** | Reading PDF, DOCX, and CSV/TXT files |

---

## Step-by-step: what happens in each stage

### Step 1: Set up the GPU

Open a new Colab notebook, switch the runtime to **T4 GPU**, and run `!nvidia-smi` to confirm a Tesla T4 with about 15 GB of memory is available.

**Why:** QLoRA needs a GPU. The free T4 is enough because the model is quantized to 4-bit and only a small set of adapter weights is trained.

### Step 2: Install Unsloth

```python
!pip install -q unsloth
import unsloth
```

**What happens:** Unsloth installs along with its dependencies and patches the model code to train about 2x faster with much less memory. This is the slowest step (several minutes), but it only happens once per session. A warning about `xformers` may appear, which is harmless because PyTorch attention is used instead.

### Step 3: Install parsers and add your documents

```python
!pip install -q pypdf python-docx pandas datasets
```

Create a folder called `docs` in the Colab sidebar and drag your files into it. The sidebar upload is faster and more reliable than `files.upload()`.

**Why:** each file type needs its own parser. `pypdf` reads PDFs, `python-docx` reads Word files, and `pandas` reads CSVs.

### Step 4: Convert documents into a dataset

This is the core of the multi-format ingestion pipeline.

```python
def read_file(path):
    ext = path.split(".")[-1].lower()
    if ext == "pdf":
        reader = PdfReader(path)
        return [p.extract_text() or "" for p in reader.pages]
    if ext == "docx":
        doc = Document(path)
        return ["\n".join(p.text for p in doc.paragraphs if p.text)]
    if ext == "txt":
        return [open(path, encoding="utf-8", errors="ignore").read()]
    if ext == "csv":
        df = pd.read_csv(path).fillna("")
        return [" | ".join(f"{c}: {r[c]}" for c in df.columns) for _, r in df.iterrows()]
    return []
```

What each part does:

- **`read_file`** picks the right parser based on the file extension and returns a list of text pieces (pages for PDFs, paragraphs for DOCX, rows for CSV).
- **`chunk`** removes extra whitespace and splits the text into pieces of about 1000 characters, dropping tiny fragments under 100 characters. Chunking keeps each training example short enough to fit in memory.
- The chunks are stored as `{"text": ..., "source": filename}` rows and turned into a Hugging Face `Dataset`.

My run produced **751 training chunks** from a single PDF. Unsupported files and hidden folders such as `.ipynb_checkpoints` are skipped automatically.

### Step 5: Load the model and attach LoRA

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    max_seq_length = 1024,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
```

**What happens:**

- The base model is loaded in **4-bit quantization** (the "Q" in QLoRA), which shrinks its memory use so it fits on the free GPU.
- **LoRA adapters** are attached to the attention and MLP layers. The original model weights stay frozen, and only the small adapter matrices are trained.
- Result: **11.27 million trainable parameters out of 1.25 billion (0.90%)**.

### Step 6: Prepare the data and the trainer

```python
EOS = tokenizer.eos_token
dataset = dataset.map(lambda x: {"text": x["text"] + EOS})

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = True,
        bf16 = False,
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)
```

**What happens:** an end-of-sequence token is added to every chunk so the model learns where a text piece ends. Then the `SFTTrainer` is configured. The T4 does not support bfloat16, so `fp16` is used.

### Step 7: Train

```python
trainer_stats = trainer.train()
```

**What happens:** the model reads the chunks in batches and updates only the LoRA adapter weights. With a batch size of 2 and gradient accumulation of 4, each update effectively uses 8 examples. The first 60 steps took about 1.4 minutes, with loss falling from **5.25 to 3.96**.

### Step 8: Train longer

```python
trainer.args.max_steps = 200
trainer_stats = trainer.train()
```

**What happens:** training continues up to 200 steps (roughly two passes over the data). Loss dropped from about 3.9 to about **2.5**. The goal is not zero loss, since that would only mean the model memorized the text.

### Step 9: Test the model

```python
FastLanguageModel.for_inference(model)
prompt = "In an AI system design interview, the first step is to"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=120, use_cache=True)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

**What happens:** `for_inference` switches on Unsloth's fast generation mode. The model continues the prompt using vocabulary learned from the documents. With default settings, a small model can repeat itself, which the next step fixes.

### Step 10: Save the adapter

```python
model.save_pretrained("my_lora_adapter")
tokenizer.save_pretrained("my_lora_adapter")
```

**What happens:** only the LoRA adapter is saved, not the whole model. The key file is `adapter_model.safetensors`, which is small and easy to share.

### Step 11: Back up to Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
!cp -r my_lora_adapter /content/drive/MyDrive/my_lora_adapter
```

**Why:** Colab's local storage is wiped when the session ends. Copying the adapter to Drive keeps it safe.

### Step 12: Fix repetition and compare with the base model

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
)
```

**What happens:** sampling settings and a repetition penalty remove the looping text. The before/after comparison turns the adapter off with `model.disable_adapter()` to get the base model's answer, then turns it back on to get the fine-tuned answer for the same prompt.

---

## Results

| Metric | Value |
| --- | --- |
| Base model | Llama-3.2-1B-Instruct (4-bit) |
| Trainable parameters | 11.27M of 1.25B (0.90%) |
| Training chunks | 751 |
| Training steps | 200 |
| Loss | 5.25 -> about 2.5 |
| GPU | Free Colab Tesla T4 |
| Training time | Under 5 minutes in total |

**Prompt:** `The most important thing to evaluate in an AI system is`

**Base model:** talks generally about the "consistency" of outputs and gives a generic chatbot example.

**Fine-tuned model:** talks about latency and cost per request, reliability, and how users experience the system, matching the language of the source document.

---

## How to run it

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Go to **Runtime -> Change runtime type -> T4 GPU**.
3. Clone this repo, or upload the notebook:
   ```bash
   git clone https://github.com/ezhilarasi0509/DoctTune.git
   ```
4. Create a `docs` folder in the Colab sidebar and drop in your own `.pdf`, `.docx`, `.txt`, or `.csv` files.
5. Run the notebook cells from top to bottom.
6. Your LoRA adapter is saved to `my_lora_adapter` and copied to Google Drive.

---

## Project structure

```
DoctTune/
|-- DocTune.ipynb      # the full Colab notebook (all steps above)
|-- README.md          # this file
|-- docs/              # put your own input documents here
`-- my_lora_adapter/   # saved LoRA adapter (output of training)
```

> Note: only add documents you have the right to share. If you train on a published book or other copyrighted material, keep the source file out of the public repo.

---

## Settings explained

| Setting | Value | Meaning |
| --- | --- | --- |
| `load_in_4bit` | True | Quantizes the model to save GPU memory (QLoRA) |
| `max_seq_length` | 1024 | Longest input in tokens |
| `r` | 16 | LoRA rank: higher means more capacity but more memory |
| `lora_alpha` | 16 | LoRA scaling factor |
| `learning_rate` | 2e-4 | Standard rate for LoRA training |
| `max_steps` | 200 | Total training steps |
| `optim` | adamw_8bit | Memory-saving optimizer |
| `repetition_penalty` | 1.2 | Reduces looping text at generation time |

---

## Limitations

- **Raw text training:** the model was trained on plain text chunks, so it continues text in the document's style rather than answering questions like a chatbot.
- **PDF extraction quality:** some words are merged together (for example, "thepercentageofquestions"), because PDF text extraction is not perfect.
- **Small model and short training:** a 1B model trained for 200 steps picks up vocabulary and style, but it is not reliable for factual accuracy.
- **No formal evaluation:** results were checked by reading sample outputs, not with a benchmark.

---

## Future improvements

- Convert chunks into **question-and-answer pairs** (instruction format) so the model can answer questions directly.
- Clean PDF text (fix merged words, remove headers and footers).
- Add a **validation split** and track validation loss.
- Export to **GGUF** to run the model locally with tools like Ollama or llama.cpp.
- Push the adapter to the **Hugging Face Hub**.
- Try a larger model such as Llama-3.2-3B or Llama-3.1-8B.

---

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) for fast, free fine-tuning
- [Hugging Face](https://huggingface.co/) for Transformers, TRL, PEFT, and Datasets
- Meta for the Llama 3.2 model
