class EightBitAdder:
    def __init__(self):
        # Initialize the 8-bit sum (0-255) and magnitude counter
        self.sum = 0
        self.magnitude = 0
    
    def add(self, value):
        # For large values, we need to handle them specially
        if value >= 256:
            # Calculate how many full 256-blocks to add to magnitude
            full_blocks = value // 256
            self.magnitude = min(self.magnitude + full_blocks, 255)
            # Only add the remainder to sum
            value = value % 256
        
        # Store old sum to check for overflow
        old_sum = self.sum
        
        # Add value and keep only the lowest 8 bits (modulo 256)
        self.sum = (self.sum + value) % 256
        
        # Check for overflow and increment magnitude
        if self.sum < old_sum:
            # Increment magnitude but keep it within 8 bits
            self.magnitude = min(self.magnitude + 1, 255)
    
    def get_bounds(self):
        # Lower bound is magnitude * 256 + sum
        lower_bound = (self.magnitude * 256) + self.sum
        
        # Upper bound is (magnitude + 1) * 256 - 1
        upper_bound = ((self.magnitude + 1) * 256) - 1
        
        return (lower_bound, upper_bound)
    
    def __str__(self):
        lower, upper = self.get_bounds()
        return f"Sum is between {lower} and {upper} (8-bit sum: {self.sum}, magnitude: {self.magnitude})"


# Example usage
if __name__ == "__main__":
    adder = EightBitAdder()
    
    # Add some values
    print("Starting state:", adder)
    
    adder.add(100)
    print("After adding 100:", adder)
    
    adder.add(200)
    print("After adding 200:", adder)
    
    # This should cause overflow
    adder.add(200)
    print("After adding another 200:", adder)
    
    # Add a large number
    adder.add(1000)
    print("After adding 1000:", adder)
