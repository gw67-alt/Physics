/*
 * Arduino Surjection Checker
 * 
 * This sketch measures discrete analog values from two pins over time,
 * stores them as sets, and determines if there's a surjection from set A to set B.
 * 
 * A surjection (onto function) exists if every element in set B
 * has at least one corresponding element in set A that maps to it.
 */

const int analogPinA = A0;    // Analog input pin for first set of values
const int analogPinB = A1;    // Analog input pin for second set of values
const int buttonPin = 2;      // Digital pin for button to start/stop recording
const int ledPin = 13;        // LED to indicate recording status

const int maxSamples = 100;   // Maximum number of samples to store
const int discretizeLevels = 10; // Number of discrete levels to map analog values to

// Arrays to store discretized values from both analog inputs
int valuesA[maxSamples];
int valuesB[maxSamples];
int sampleCount = 0;
bool isRecording = false;

// Arrays to store unique values found in each set
int uniqueValuesA[discretizeLevels];
int uniqueCountA = 0;
int uniqueValuesB[discretizeLevels];
int uniqueCountB = 0;

// Mapping array to check surjection
int mappingMatrix[discretizeLevels][discretizeLevels]; // [A][B]

void setup() {
  Serial.begin(9600);
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(ledPin, OUTPUT);
  
  Serial.println("Arduino Surjection Checker");
  Serial.println("Press button to start/stop recording values");
  Serial.println("Results will be analyzed after recording stops");
}

void loop() {
  // Check if button is pressed to toggle recording
  if (digitalRead(buttonPin) == LOW) {
    delay(50); // Debounce
    if (digitalRead(buttonPin) == LOW) {
      while (digitalRead(buttonPin) == LOW); // Wait for release
      toggleRecording();
    }
  }
  
  // If recording and not at max samples, read values
  if (isRecording && sampleCount < maxSamples) {
    // Read analog values
    int rawA = analogRead(analogPinA);
    int rawB = analogRead(analogPinB);
    
    // Discretize values (map 0-1023 to 0-(discretizeLevels-1))
    int discreteA = map(rawA, 0, 1023, 0, discretizeLevels - 1);
    int discreteB = map(rawB, 0, 1023, 0, discretizeLevels - 1);
    
    // Store the values
    valuesA[sampleCount] = discreteA;
    valuesB[sampleCount] = discreteB;
    
    // Print current reading
    Serial.print("Sample ");
    Serial.print(sampleCount + 1);
    Serial.print(": A=");
    Serial.print(discreteA);
    Serial.print(", B=");
    Serial.println(discreteB);
    
    sampleCount++;
    
    // Blink LED to indicate recording
    digitalWrite(ledPin, HIGH);
    delay(100);
    digitalWrite(ledPin, LOW);
    delay(400);  // Take a sample roughly every 500ms
  }
  
  // If we've reached max samples, stop recording
  if (isRecording && sampleCount >= maxSamples) {
    toggleRecording();
    Serial.println("Maximum samples reached. Recording stopped.");
  }
}

void toggleRecording() {
  isRecording = !isRecording;
  
  if (isRecording) {
    // Start a new recording session
    sampleCount = 0;
    Serial.println("Recording started...");
    digitalWrite(ledPin, HIGH);
  } else {
    // Stop recording and analyze results
    Serial.println("Recording stopped.");
    digitalWrite(ledPin, LOW);
    analyzeResults();
  }
}

void analyzeResults() {
  if (sampleCount == 0) {
    Serial.println("No samples recorded. Nothing to analyze.");
    return;
  }
  
  Serial.print("Analyzing ");
  Serial.print(sampleCount);
  Serial.println(" samples...");
  
  // Reset analysis variables
  uniqueCountA = 0;
  uniqueCountB = 0;
  memset(uniqueValuesA, -1, sizeof(uniqueValuesA));
  memset(uniqueValuesB, -1, sizeof(uniqueValuesB));
  memset(mappingMatrix, 0, sizeof(mappingMatrix));
  
  // Find unique values in both sets and build mapping matrix
  for (int i = 0; i < sampleCount; i++) {
    int valueA = valuesA[i];
    int valueB = valuesB[i];
    
    // Add to unique values in A if not already there
    if (!contains(uniqueValuesA, uniqueCountA, valueA)) {
      uniqueValuesA[uniqueCountA] = valueA;
      uniqueCountA++;
    }
    
    // Add to unique values in B if not already there
    if (!contains(uniqueValuesB, uniqueCountB, valueB)) {
      uniqueValuesB[uniqueCountB] = valueB;
      uniqueCountB++;
    }
    
    // Record this mapping in the matrix
    mappingMatrix[valueA][valueB]++;
  }
  
  // Print unique values
  Serial.print("Set A has ");
  Serial.print(uniqueCountA);
  Serial.print(" unique values: ");
  printArray(uniqueValuesA, uniqueCountA);
  
  Serial.print("Set B has ");
  Serial.print(uniqueCountB);
  Serial.print(" unique values: ");
  printArray(uniqueValuesB, uniqueCountB);
  
  // Check if there's a surjection from A to B
  bool isSurjection = true;
  for (int j = 0; j < uniqueCountB; j++) {
    int valueB = uniqueValuesB[j];
    bool hasPreimage = false;
    
    for (int i = 0; i < uniqueCountA; i++) {
      int valueA = uniqueValuesA[i];
      if (mappingMatrix[valueA][valueB] > 0) {
        hasPreimage = true;
        break;
      }
    }
    
    if (!hasPreimage) {
      isSurjection = false;
      Serial.print("Value ");
      Serial.print(valueB);
      Serial.println(" in set B has no corresponding value in set A");
    }
  }
  
  // Print the conclusion
  if (isSurjection) {
    Serial.println("RESULT: There IS a surjection from set A to set B");
  } else {
    Serial.println("RESULT: There is NO surjection from set A to set B");
  }
  
  // Print mapping details
  Serial.println("\nMapping details:");
  for (int i = 0; i < uniqueCountA; i++) {
    int valueA = uniqueValuesA[i];
    Serial.print("A=");
    Serial.print(valueA);
    Serial.print(" maps to: ");
    
    bool firstMapping = true;
    for (int j = 0; j < uniqueCountB; j++) {
      int valueB = uniqueValuesB[j];
      if (mappingMatrix[valueA][valueB] > 0) {
        if (!firstMapping) {
          Serial.print(", ");
        }
        Serial.print("B=");
        Serial.print(valueB);
        Serial.print("(");
        Serial.print(mappingMatrix[valueA][valueB]);
        Serial.print(" times)");
        firstMapping = false;
      }
    }
    Serial.println();
  }
}

// Helper function to check if an array contains a value
bool contains(int arr[], int size, int value) {
  for (int i = 0; i < size; i++) {
    if (arr[i] == value) {
      return true;
    }
  }
  return false;
}

// Helper function to print an array
void printArray(int arr[], int size) {
  Serial.print("{");
  for (int i = 0; i < size; i++) {
    Serial.print(arr[i]);
    if (i < size - 1) {
      Serial.print(", ");
    }
  }
  Serial.println("}");
}
