# Problem Statement – Deep Learning (Transliteration Task)

The task is to build a **character-level neural transliteration model** that converts words written in **Roman (Latin) script** into their corresponding **native-script** forms using the **Aksharantar dataset**.  

Each data pair consists of a romanized word and its native-script equivalent:  
> Example: `ghar → घर`

---

##  Objective
Design and train a **sequence-to-sequence (Seq2Seq)** model using **RNN-based architectures** that can automatically learn the mapping between characters in different scripts.

---

##  Model Architecture

The model includes the following key components:

1. **Embedding Layer** – Converts each character into a dense vector representation.  
2. **Encoder RNN (LSTM/GRU/RNN)** – Processes the input character sequence and encodes it into a fixed-length context vector.  
3. **Decoder RNN** – Generates one output character at a time using the encoder’s context and previously predicted characters.

---

##  Implementation Details

- Implemented in **Python** using frameworks like **PyTorch** or **TensorFlow**.  
- Should be **runnable on Google Colab or Kaggle** with **GPU acceleration**.  
- The implementation must be **flexible** in terms of:
  - Embedding dimension  
  - Hidden size  
  - RNN type (LSTM / GRU / Vanilla RNN)  
  - Number of layers  

- Maintained in a **GitHub repository** following proper version control and documentation practices.

---

##  Additional Task

Compute the **total number of computations and parameters** in the network under simplified assumptions:
- Single-layer encoder and decoder  
- Equal input and output sequence lengths  
- Same vocabulary size for both source and target languages  

---

##  Dataset
**Aksharantar Dataset**  
A multilingual transliteration dataset containing pairs of Romanized and native-script words across several Indian languages.

---

##  Expected Outcome
A trained Seq2Seq transliteration model capable of converting words from Roman script to native script (e.g., Hindi, Marathi, Tamil, etc.) with high accuracy.


---

##  References

To apply the concept of a **character-level sequence-to-sequence model** for transliteration, we referred to the following key research works and incorporated their methodologies into our implementation:

---

###  1. *Romanized to Native Malayalam Script Transliteration Using an Encoder-Decoder Framework*  
**Authors:** Baiju et al.  

This paper inspired our **Encoder-Decoder framework** for transliteration.  
In our implementation:  
- We used a **Bi-LSTM encoder** (`Encoder` class with `bidir=True`).  
- A **Luong Attention** module was implemented in the decoder to enhance focus on relevant input characters during decoding.  
- This approach allowed our model to effectively map Romanized text to native-script equivalents.

---

###  2. *Sinhala Transliteration: A Comparative Analysis Between Rule-based and Seq2Seq Approaches*  
**Authors:** De Mel et al., *ACL Anthology*  

This work guided our **multilingual transliteration strategy**.  
We followed the idea of:
- **Prefixing each input** with a language tag (e.g., `<lang>`).  
- **Merging datasets** across multiple languages.  
This enabled our model to handle **multiple scripts in a unified transliteration system**, conditioning the predictions based on language context.

---

###  3. *Efficient Machine Translation with a BiLSTM-Attention Approach*  
**Authors:** Wu & Xing  

This paper influenced our design for **efficient sequence modeling**.  
Our transliteration model mirrors this architecture by:
- Using a **Bidirectional LSTM encoder** (`make_rnn(..., bidir=True)`), and  
- An **Attention-based decoder** (`DecoderWithAttention`).  

This design ensures **parameter efficiency** while maintaining **high transliteration accuracy** across diverse language pairs.

---

*Together, these research works guided the design choices for our Seq2Seq transliteration model, helping achieve a balance between efficiency, accuracy, and multilingual adaptability.*

# Deep Learning Project

---

##  Getting Started

##### Follow these steps to clone the repository, set up the environment, and run the project.

---

### 🚀 Setup Instructions

#### 1. Clone the Repository
You can clone this repository using the following command:

```bash
git clone https://github.com/Savale-pandurang/multilingual_transliteration.git
```
#### 2. Navigate to the Project Directory

Change your current directory to the newly cloned project folder:
```
cd multilingual_transliteration
```
#### 3. Install required dependencies
```
pip install -r requirements.txt
```
#### 4. Run the project
```
python3 main.py
```