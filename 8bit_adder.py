class EightBitNumberDisplay:
    def __init__(self):
        # Use 5 bits for mantissa and 3 bits for exponent
        self.mantissa_bits = 5
        self.exponent_bits = 3
        self.mantissa = 0  # 5 bits (0-31)
        self.exponent = 0  # 3 bits (0-7)
    
    def store_number(self, number):
        if number == 0:
            self.mantissa = 0
            self.exponent = 0
            return
            
        # Find the highest bit position that is set
        bit_position = 0
        temp = number
        while temp > 0:
            temp >>= 1
            bit_position += 1
        
        # Calculate needed exponent (how many bits to shift)
        # We want to keep the 5 most significant bits
        if bit_position <= self.mantissa_bits:
            self.exponent = 0
            self.mantissa = number
        else:
            # Calculate how many bits to shift right
            shift = bit_position - self.mantissa_bits
            # Limit exponent to max value (7 for 3 bits)
            if shift > (2**self.exponent_bits - 1):
                self.exponent = (2**self.exponent_bits) - 1
                self.mantissa = (number >> (shift - self.exponent)) & ((1 << self.mantissa_bits) - 1)
            else:
                self.exponent = shift
                # Extract mantissa (most significant bits)
                self.mantissa = (number >> shift) & ((1 << self.mantissa_bits) - 1)
    
    def get_value(self):
        # For exponent 0, return mantissa directly
        if self.exponent == 0:
            return self.mantissa
        else:
            # Shift mantissa left by exponent
            return self.mantissa << self.exponent
    
    def __str__(self):
        value = self.get_value()
        if self.exponent == (2**self.exponent_bits - 1):
            return f"~{value} (approximate, max exponent reached)"
        else:
            return f"{value} (mantissa: {self.mantissa}, exponent: {self.exponent})"


# Example usage
if __name__ == "__main__":
    display = EightBitNumberDisplay()
    
    # Test with different sizes of numbers
    test_numbers = [0, 10, 31, 32, 100, 500, 1000, 10000, 100000, 1000000]
    
    for num in test_numbers:
        display.store_number(num)
        print(f"Original: {num}, Stored: {display}")
