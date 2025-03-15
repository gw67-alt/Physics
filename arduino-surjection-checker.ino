/*
 * Arduino Non-injective Surjective Analyzer with Threshold Triggering
 * 
 * This sketch waits for both analog values to exceed a threshold, then automatically
 * measures discrete analog values from two pins, and stops after collecting
 * the specified number of samples.
 * It analyzes whether the mapping from set A to set B is:
 * 1. Surjective (onto): every element in B has at least one corresponding element in A
 * 2. Non-injective: at least one element in B has multiple elements from A mapping to it
 * 
 * After analysis, it returns to the waiting state, ready for another trigger event.
 */

const int analogPinA = A0;    // Analog input pin for first set of values
const int analogPinB = A1;    // Analog input pin for second set of values
const int ledPin = 13;        // LED to indicate recording status

const int maxSamples = 32;    // Maximum number of samples to store
const int discretizeLevels = 10; // Number of discrete levels to map analog values to
const int sampleInterval = 35; // Time between samples in milliseconds
const int signalThreshold = 680; // Threshold value (0-1023) that both signals must exceed to start

// Arrays to store discretized values from both analog inputs
int valuesA[maxSamples];
int valuesB[maxSamples];
int sampleCount = 0;
bool analysisComplete = false;
bool waitingForTrigger = true;

// Arrays to store unique values found in each set
int uniqueValuesA[discretizeLevels];
int uniqueCountA = 0;
int uniqueValuesB[discretizeLevels];
int uniqueCountB = 0;

// Mapping array to check properties
int mappingMatrix[discretizeLevels][discretizeLevels]; // [A][B]

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
  
  Serial.println("Arduino Non-injective Surjective Analyzer with Threshold Triggering");
  Serial.println("Waiting for both analog signals to exceed threshold...");
}

void loop() {
  // Check for trigger condition if we're waiting for it
  if (waitingForTrigger) {
    // Read raw analog values
    int rawA = analogRead(analogPinA);
    int rawB = analogRead(analogPinB);
    
    // Slow blink LED to indicate waiting state
    slowBlinkLED();
    
    // Check if both values exceed threshold
    if (rawA > signalThreshold && rawB > signalThreshold) {
      Serial.print("Trigger detected! A=");
      Serial.print(rawA);
      Serial.print(", B=");
      Serial.println(rawB);
      
      // Reset sample collection variables
      sampleCount = 0;
      analysisComplete = false;
      waitingForTrigger = false;
      
      // Set LED solid on to indicate recording will start
      digitalWrite(ledPin, HIGH);
      Serial.println("Starting sample collection...");
      
      // Small delay to debounce and prepare
      delay(100);
    }
    return; // Skip the rest of the loop while waiting for trigger
  }

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
    delay(10);
    digitalWrite(ledPin, HIGH);
    delay(sampleInterval - 10);  // Account for the blink time
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
    
    // Reset to wait for next trigger
    waitingForTrigger = true;
    Serial.println("\nReady for next trigger event. Waiting for both analog signals to exceed threshold...");
  }
}

void slowBlinkLED() {
  // Slow blink pattern for waiting state
  static unsigned long lastToggleTime = 0;
  static boolean ledState = false;
  unsigned long currentTime = millis();
  
  if (currentTime - lastToggleTime > 500) {  // 0.5 second toggle
    ledState = !ledState;
    digitalWrite(ledPin, ledState);
    lastToggleTime = currentTime;
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
  
  // Check surjectivity (onto property)
  bool isSurjective = true;
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
      isSurjective = false;
      Serial.print("Value B=");
      Serial.print(valueB);
      Serial.println(" has no corresponding value in set A (not surjective)");
    }
  }
  
  // Check non-injectivity (checking for many-to-one mappings)
  bool isNonInjective = false;
  for (int j = 0; j < uniqueCountB; j++) {
    int valueB = uniqueValuesB[j];
    int preimageCount = 0;
    
    for (int i = 0; i < uniqueCountA; i++) {
      int valueA = uniqueValuesA[i];
      if (mappingMatrix[valueA][valueB] > 0) {
        preimageCount++;
      }
    }
    
    if (preimageCount > 1) {
      isNonInjective = true;
      Serial.print("Value B=");
      Serial.print(valueB);
      Serial.print(" has ");
      Serial.print(preimageCount);
      Serial.println(" different values from set A mapping to it (non-injective)");
    }
  }
  
  // Print function property conclusions
  Serial.println("\n--- FUNCTION PROPERTY ANALYSIS ---");
  if (isSurjective && isNonInjective) {
    Serial.println("RESULT: The mapping IS Non-injective and Surjective (many-to-one and onto)");
  } else if (isSurjective && !isNonInjective) {
    Serial.println("RESULT: The mapping is Injective and Surjective (bijective/one-to-one and onto)");
    Serial.println("  This is NOT the non-injective surjective mapping we're looking for");
  } else if (!isSurjective && isNonInjective) {
    Serial.println("RESULT: The mapping is Non-injective but NOT Surjective (many-to-one but not onto)");
    Serial.println("  This is NOT the non-injective surjective mapping we're looking for");
  } else {
    Serial.println("RESULT: The mapping is neither Non-injective nor Surjective");
    Serial.println("  This is NOT the non-injective surjective mapping we're looking for");
  }
  
  // Print detailed mapping information
  Serial.println("\nMapping details:");
  // For each unique value in A, show what it maps to
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
        Serial.print(" (");
        Serial.print(mappingMatrix[valueA][valueB]);
        Serial.print(" times)");
        firstMapping = false;
      }
    }
    Serial.println();
  }
  
  // For each unique value in B, show what maps to it
  Serial.println("\nReverse mapping details:");
  for (int j = 0; j < uniqueCountB; j++) {
    int valueB = uniqueValuesB[j];
    Serial.print("B=");
    Serial.print(valueB);
    Serial.print(" has preimages: ");
    
    bool firstMapping = true;
    int preimageCount = 0;
    for (int i = 0; i < uniqueCountA; i++) {
      int valueA = uniqueValuesA[i];
      if (mappingMatrix[valueA][valueB] > 0) {
        if (!firstMapping) {
          Serial.print(", ");
        }
        Serial.print("A=");
        Serial.print(valueA);
        firstMapping = false;
        preimageCount++;
      }
    }
    
    if (preimageCount == 0) {
      Serial.print("none (affects surjectivity)");
    } else if (preimageCount == 1) {
      Serial.print(" (injective for this value)");
    } else {
      Serial.print(" (non-injective for this value)");
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
