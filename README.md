### **Mental Health Condition Detector from Text**

#### **Objective**
The project focuses on building a machine learning model to detect mental health conditions based on textual inputs. By leveraging advanced natural language processing (NLP) techniques and neural networks, the goal is to classify text into appropriate mental health condition categories.

---

### **Steps and Techniques Implemented**

#### **1. Preprocessing**
- **Data Handling**: Text data from the `title` and `content` columns were combined into a single field, ensuring comprehensive analysis of the input.
- **Missing Data Management**: Missing values in the `content` column were handled by replacing them with empty strings.
- **Label Encoding**: The categorical labels for mental health conditions were converted into numerical format using the `LabelEncoder`.

#### **2. Tokenization**
- Used Hugging Face's `BertTokenizer` for text tokenization:
  - Converted the text into tokens compatible with the `bert-base-uncased` model.
  - Applied padding and truncation to ensure uniform input sizes (maximum length set to 128 tokens).
  - Generated attention masks to denote meaningful tokens versus padding.

#### **3. Data Augmentation**
- Implemented a simple data augmentation method:
  - Shuffled words in the text to create diverse training examples.
  - Augmented data helps improve the model's robustness and reduce overfitting.

#### **4. Dataset Preparation**
- Split the dataset into training and validation sets using an 80-20 ratio.
- Created TensorFlow datasets (`tf.data.Dataset`) for efficient batching and shuffling.
- Combined the original training dataset with the augmented dataset to enhance model training.

#### **5. Model Architecture**
- Utilized the pre-trained BERT model (`bert-base-uncased`) for feature extraction.
- Added a custom classification head:
  - A dropout layer with a rate of 0.5 to prevent overfitting.
  - A dense layer with a softmax activation function for multi-class classification.
- The architecture is designed to balance pre-trained knowledge with task-specific learning.

#### **6. Training Procedure**
- **Cross-Validation**: Performed 5-fold cross-validation using `KFold` from Scikit-learn:
  - Split the data into five subsets, training on four and validating on one in a rotating manner.
  - Evaluated the model's generalization performance across different data splits.
- **Optimization**:
  - Used the Adam optimizer with a very small learning rate (`1e-6` initially).
  - Implemented `SparseCategoricalCrossentropy` as the loss function, suitable for multi-class classification.

#### **7. Callbacks for Training Optimization**
- **ReduceLROnPlateau**: Reduced the learning rate when validation loss plateaued, aiding fine-tuning.
- **EarlyStopping**: Stopped training early when no improvement in validation loss was observed for three consecutive epochs.
- **LearningRateScheduler**: Dynamically adjusted the learning rate based on the epoch number.

#### **8. Performance Evaluation**
- Fold-wise validation accuracy:
  - Fold 1: **74.18%**
  - Fold 2: **78.24%**
  - Fold 3: **80.28%**
  - Fold 4: **80.78%**
  - Fold 5: **82.28%**
- Monitored the model's performance across folds, ensuring consistent improvement and reduced overfitting.

#### **9. Insights Gained**
- Validation accuracy steadily improved as the model learned from augmented and original data.
- Training loss decreased consistently, demonstrating effective optimization of weights.
- The cross-validation framework ensured robustness by exposing the model to diverse subsets of the dataset.
