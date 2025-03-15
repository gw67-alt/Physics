/*
 * Arduino Surjection Checker with Auto Start/Stop
 * 
 * This sketch automatically starts measuring discrete analog values from two pins when powered up,
 * and automatically stops after collecting the specified number of samples.
 * It then analyzes whether there's a surjection from set A to set B.
 * 
 * A surjection (onto function) exists if every element in set B
 * has at least one corresponding element in set A that maps to it.
 */

const int analogPinA = A0;    // Analog input pin for first set of values
const int analogPinB = A1;    // Analog input pin for second set of values
const int ledPin = 13;        // LED to indicate recording status

const int maxSamples = 100;   // Maximum number of samples to store
const int discretizeLevels = 10; // Number of discrete levels to map analog values to
const int sampleInterval = 500; // Time between samples in milliseconds

// Arrays to store discretized values from both analog inputs
int valuesA[maxSamples];
int valuesB[maxSamples];
int sampleCount = 0;
bool analysisComplete = false;

// Arrays to store unique values found in each set
int uniqueValuesA[discretizeLevels];
int uniqueCountA = 0;
int uniqueValuesB[discretizeLevels];
int uniqueCountB = 0;

// Mapping array to check surjection
int mappingMatrix[discretizeLevels][discretizeLevels]; // [A][B]

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
  
  Serial.println("Arduino Surjection Checker (Auto Mode)");
  Serial.println("Automatically collecting samples...");
  
  // Set LED on to indicate recording is active
  digitalWrite(ledPin, HIGH);
}

void loop() {
  // If we haven't completed analysis and haven't collected all samples
  if (!analysisComplete && sampleCount < maxSamples) {
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
    
    // Blink LED to indicate recording activity
    digitalWrite(ledPin, LOW);
    delay(100);
    digitalWrite(ledPin, HIGH);
    delay(sampleInterval - 100);  // Account for the blink time
  }
  
  // If we've reached max samples and haven't analyzed yet
  if (!analysisComplete && sampleCount >= maxSamples) {
    // Turn off LED to indicate recording stopped
    digitalWrite(ledPin, LOW);
    Serial.println("Sample collection complete. Analyzing results...");
    
    // Analyze the results
    analyzeResults();
    
    // Mark analysis as complete
    analysisComplete = true;
    
    // Flash LED rapidly 5 times to indicate completion
    for (int i = 0; i < 5; i++) {
      digitalWrite(ledPin, HIGH);
      delay(100);
      digitalWrite(ledPin, LOW);
      delay(100);
    }
  }
  
  // If analysis is complete, just wait (or could implement a reset button here)
  if (analysisComplete) {
    // Flash LED once every 2 seconds to indicate idle state
    digitalWrite(ledPin, HIGH);
    delay(200);
    digitalWrite(ledPin, LOW);
    delay(1800);
  }
}

void analyzeResults() {
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
