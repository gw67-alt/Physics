
        
        print("\nOriginal Parameters:")
        print(f"  Desired Atoms per period: {quantized_result['crystal_dimension']['atoms']}")
        print(f"  Periodicity: {periodicity}")
        
        print("\nCrystal Dimension:")
        print(f"  Total Dimension: {quantized_result['crystal_dimension']['total_dimension_nm']:.4f} nm")
        print(f"  Lattice Constant: {quantized_result['crystal_dimension']['lattice_constant_nm']:.4f} nm")
        
        print("\nInterval Quantization:")
        print(f"  Result: {quantized_result['max_interval_frequency']:}")
        
        print("\n" + "=" * 50)

    except ValueError as ve:
        print(f"Input Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
