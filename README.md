
# 🧠 ViT-Implementation — Your Friendly Vision Transformer!

Welcome to **ViT-Implementation**, where image patches go to school, learn attention, and graduate as powerful classification wizards 🎓📸✨

This repo contains a **from-scratch implementation** of the **Vision Transformer (ViT)** architecture using PyTorch. We've built everything ourselves: patch embeddings, attention heads, transformer blocks, and more — because why use pretrained models when you can **reinvent the wheel with style**? 😎

This implementation is inspired by the groundbreaking [Vision Transformer (ViT) paper](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al., which introduced the idea of applying transformer architectures directly to image classification.

---

## 🚀 Getting Started

To begin your journey with our ViT, follow these simple spells (commands):

### 🏋️‍♀️ 1. Train the model
```bash
python -m src.train
```

This will train your Vision Transformer on the Imagenette dataset (or whatever dataset you plug in). Make sure you stretch before running this — those attention heads are heavy!

---

### 🧠 2. Evaluate the model
```bash
python -m src.evaluate
```

This will evaluate `best_model.pt` and print out the accuracy of your model. Be ready to be impressed — or to go back and tune those hyperparameters (again 😅).

---

### 📊 3. Generate a CSV of your hyperparameters & results
```bash
python -m src.csv_generator
```

Want to keep track of your experimentation madness? This command will generate a neat CSV so you can compare models like a proper deep learning scientist. 🧪📈

---

## 🧩 What’s Inside?

Here's what makes this repo cool:

- ✅ **Patch Embeddings** — Slice up images into small pieces and feed them to the model like popcorn.
- ✅ **GELU Activation** — Not just ReLU... we're fancier than that.
- ✅ **Multi-Head Self-Attention** — Because one head isn't enough to understand all the image drama.
- ✅ **Transformer Encoder Blocks** — Each with residual connections and layer norms, obviously.
- ✅ **From-scratch Training Pipeline** — Built lovingly with PyTorch ❤️
- ✅ **Modular & Readable Code** — So your future self doesn’t hate you.

---

## 🤖 Requirements

Make sure you’ve got the basics installed:

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🧙‍♀️ Fun Fact

This ViT is trained on images, but deep inside it's powered by **math, magic, and motivation**. Every line of code has been written to be understandable, reusable, and cool.

---

## 📝 Credits

Made with ❤️ by Lydia Ruiz Martínez.  
Inspired by the original [ViT paper](https://arxiv.org/abs/2010.11929) and a bit of ✨ stubbornness ✨ to implement things from scratch.

---

## 🌟 Star This Repo!

If you found this helpful or just had fun reading this README, consider giving it a ⭐️. It fuels the Transformer (and the author)!
