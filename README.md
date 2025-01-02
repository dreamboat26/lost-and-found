# Euclidean Braid Encoder

## Overview
The **Euclidean Braid Encoder** is a Python-based program that encodes text messages into 2D braid-like visualizations by leveraging crossings between multiple straight-line paths. The lines and crossings are visualized in a graph, making it a unique way to combine geometry with binary data representation.

## Features
- **Path Generation**: Create straight-line paths between defined points in 2D space.
- **Crossing Detection**: Automatically detect crossings between paths based on proximity thresholds.
- **Message Encoding**: Encode text messages as binary data (0s and 1s) into detected crossings.
- **Visualization**: Visualize the lines, crossings, and encoded bits in an easy-to-understand plot.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/dreamboat26/lost-and-found.git
   cd euclidean-braid-encoder
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Example
The program comes with a main function that demonstrates how to use the encoder. To run it:

```bash
python encoder.py
```

The program will:
1. Define 7 straight-line paths.
2. Detect crossings between these paths.
3. Encode the message `"Hello World"` into the crossings.
4. Visualize the lines and the encoded binary bits on a graph.

### Output
- A plot showing:
  - Paths in different colors.
  - Detected crossings marked with binary bits (0 or 1).
- Console output with details about the encoding process.

### Customization
You can customize:
- **Number of lines**: Add or modify the paths in the `lines` list.
- **Message**: Change the message being encoded in the `main()` function.
- **Threshold**: Adjust the crossing detection threshold in `find_all_crossings()`.

## Code Structure

### `EuclideanBraidEncoder`
This is the main class that provides:

1. **Line Management**
   - `add_line(start, end)`: Add a line to the system.

2. **Crossing Detection**
   - `find_all_crossings(threshold)`: Detect and store crossings based on proximity.

3. **Message Encoding**
   - `encode_message(message)`: Convert a text message into binary and associate it with detected crossings.

4. **Visualization**
   - `visualize(encoded_braids)`: Display the braid system with optional binary annotations.

### Example Code
```python
# Initialize encoder
encoder = EuclideanBraidEncoder()

# Add lines
encoder.add_line((-1, -0.5), (1, 0.5))
encoder.add_line((-1, 0.5), (1, -0.5))

# Find crossings
crossings = encoder.find_all_crossings()

# Encode message
message = "Hello"
encoded = encoder.encode_message(message)

# Visualize
encoder.visualize(encoded)
```

## Requirements
- Python 3.7+
- Required Python libraries:
  - `matplotlib`
  - `numpy`
  - `scipy`

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features:
1. Fork the repository.
2. Create a new branch.
3. Submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
This project was inspired by the concept of encoding data into visual patterns and exploring the intersection of geometry and data science.
