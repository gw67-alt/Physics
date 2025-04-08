import sys
sys.set_int_max_str_digits(1000000000)
class EightBitNumberDisplay:
    def __init__(self):
        # Use 5 bits for mantissa and 3 bits for exponent
        self.mantissa_bits = 5000
        self.exponent_bits = 3000
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
    
    def multiply(self, other_number):
        # Convert the other number to our format
        other = EightBitNumberDisplay()
        if isinstance(other_number, EightBitNumberDisplay):
            other.mantissa = other_number.mantissa
            other.exponent = other_number.exponent
        else:
            other.store_number(other_number)
        
        # Handle zeros
        if self.mantissa == 0 or other.mantissa == 0:
            self.mantissa = 0
            self.exponent = 0
            return
        
        # Multiply mantissas
        result_mantissa = self.mantissa * other.mantissa
        
        # Add exponents
        result_exponent = self.exponent + other.exponent
        
        # Handle overflow in mantissa
        bit_position = 0
        temp = result_mantissa
        while temp > 0:
            temp >>= 1
            bit_position += 1
        
        # Adjust if mantissa overflows 5 bits
        if bit_position > self.mantissa_bits:
            shift = bit_position - self.mantissa_bits
            result_mantissa >>= shift
            result_exponent += shift
        
        # Handle exponent overflow
        if result_exponent > (2**self.exponent_bits - 1):
            result_exponent = (2**self.exponent_bits - 1)
            # We've reached maximum exponent, precision is lost
        
        # Store results
        self.mantissa = result_mantissa & ((1 << self.mantissa_bits) - 1)
        self.exponent = result_exponent
    
    def __str__(self):
        value = self.get_value()
        if self.exponent == (2**self.exponent_bits - 1):
            return f"~{value} (approximate, max exponent reached)"
        else:
            return f"{value} (mantissa: {self.mantissa}, exponent: {self.exponent})"


# Example usage with multiplication
if __name__ == "__main__":

    # Test multiplication
    print("\n=== Algorithm ===")

    
    # Example 4: Very large numbers
    num = EightBitNumberDisplay()
    num.store_number(999)
    print(f"\nStarting with: {num}")
    arg = 8
    for i in range(arg):
        for j in range(arg):
            for k in range(arg):
                for l in range(arg):
                    num.multiply(i+1)
                    num.multiply(j+1)
                    num.multiply(k+1)
                    num.multiply(l+1)
                    
    print(f"After multiplying by: {num}")
