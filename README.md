# Residual Injection for Transfer Learning

This work demonstrates the concept of **Residual Injection** for **Transfer Learning**, using neural networks trained on synthetic datasets. Residual injection involves transferring knowledge from one trained model (Task A) to another model (Task B) by injecting the residuals of the learned weights into a new model.

---

## **Table of Contents**

- [Overview](#overview)
- [Datasets Used](#datasets-used)
- [Project Workflow](#project-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)

---

## **Overview**

This project investigates how knowledge learned from one task (e.g., classifying interleaved moons) can be transferred to another task (e.g., classifying concentric circles). By injecting residuals from the weights of a trained model into a new model, we aim to accelerate learning or improve performance on the second task.

Key techniques explored include:

- **Residual Injection**: Transferring the difference in weights between trained and random models.
- **Transfer Learning**: Using prior knowledge from Task A to improve learning in Task B.

---

## **Datasets Used**

1. **Task A: Moons Dataset**
   - Two interleaved half-circles (moons).
   - Used as the source task to train the initial model.

2. **Task B: Circles Dataset**
   - Two concentric circles.
   - Used as the target task to test the impact of residual injection.

---

## **Project Workflow**

1. **Train Model on Task A**:
   - Train a neural network on the moons dataset.
   - Save the trained model's weights.

2. **Prepare Task B**:
   - Generate the circles dataset.
   - Create two models: a blank model and an injected model.

3. **Inject Residuals**:
   - Calculate residuals (differences in weights) from Task A.
   - Inject these residuals into the blank model for Task B.

4. **Train and Compare**:
   - Train both the blank and injected models on Task B.
   - Compare their accuracy and learning speed.

5. **Visualize Results**:
   - Plot training accuracy over epochs.
   - Highlight the differences between the two models.

---

## **Installation**

To run this project, ensure you have the following dependencies installed:

```bash
pip install numpy matplotlib scikit-learn tensorflow
```

---

## **Usage**

1. Clone this repository:
   ```bash
   git clone https://github.com/dreamboat26/lost-and-found.git
   cd Residual-Injection-for-Transfer-Learning
   ```
2. Visualize the results and review the output accuracies for both models.

---

## **Results**

The injected model is expected to perform better or converge faster, demonstrating the effectiveness of residual injection.

---

## **Visualizations**

### Training Accuracy Comparison

The training accuracy for both the blank model and the injected model over 10 epochs. The injected model typically outperforms the blank model.

---

## **Contributing**

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

