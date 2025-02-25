# Experimental Configuration Guidelines

This document provides detailed instructions for setting up the Negative Time Machine experimental apparatus. Following these guidelines will ensure consistent results and proper operation of all components.

## Laboratory Requirements

### Environment

- **Temperature**: 20±1°C, stable to within 0.1°C during experiments
- **Humidity**: 30-50%, non-condensing
- **Vibration Isolation**: Pneumatic optical table with >90% isolation efficiency at >5Hz
- **EMI Shielding**: Faraday cage surrounding critical components; minimum 60dB attenuation at 1GHz
- **Ambient Light**: Blackout conditions (<0.01 lux) during experimental runs

### Space Requirements

- Minimum 3m × 3m dedicated clean laboratory space
- 2m × 1.5m optical table area
- Ceiling clearance: 2.5m minimum

## Component Setup

### 1. Optical Breadboard Configuration

#### Base Installation

1. Place the optical breadboard (1m × 1m minimum) on a vibration-isolated table
2. Level using the integrated bubble level, adjusting feet until centered
3. Secure breadboard to table using provided clamps, torqued to 2.5Nm

#### Grid Layout

1. Using the reference layout diagram, mark component positions with temporary adhesive labels
2. Install all primary optical mounts, leaving components themselves uninstalled
3. Verify all mount positions with a precision ruler, tolerance ±0.5mm

### 2. Photon Source Installation

#### Laser Setup

1. Mount laser source at position A1 on the breadboard
2. Secure with M6 screws, torqued to 2.0Nm
3. Connect power supply via shielded cables only
4. Install thermal management system if using >2W source

#### Alignment

1. Use laser alignment tool to establish the primary beam path
2. Mark beam path with temporary indicators
3. Verify beam stability over 1 hour period (drift <0.1mrad)

### 3. Beam Splitter Configuration

1. Mount beam splitter at position B3, 45° to incident beam
2. Use precision rotation mount for exact angular positioning
3. Verify transmitted and reflected beam paths with alignment tool
4. Measure split ratio and adjust if not 50:50 (±2%)

### 4. CO₂ Mirror Mounting

1. Install mirror mount at position C2, using kinematic controls
2. Place CO₂ mirror in mount, securing with minimal pressure to avoid distortion
3. Connect thermal regulation system and verify temperature stability (±0.05°C)
4. Align to produce exact retroreflection of incident beam

### 5. Mechanical Blocking Oscillator Setup

#### Mechanical Assembly

1. Mount oscillator base plate at position D4
2. Install piezoelectric actuator, securing with non-magnetic fasteners
3. Attach blocking medium to actuator using the supplied mounting adapter
4. Install diffraction grating (1200 lines/mm) onto the blocking medium:
   - Use the precision alignment jig to ensure grating lines are perpendicular to beam path
   - Secure grating with cyanoacrylate adhesive applied at corners only
   - Allow 24 hours curing time before operation
   - Verify grating surface is free of dust particles using microscope inspection
5. Verify blocking medium with grating intersects beam path completely when activated

#### Electronic Connection

1. Connect oscillator to driver circuit using twisted pair shielded cable
2. Ground all shielding to central ground point
3. Connect function generator to driver input
4. Calibrate oscillator frequency and amplitude at the control console

### 6. Photodiode Installation

1. Mount primary photodiode at position E2
2. Mount secondary photodiode at position E5
3. Align both to maximize signal when oscillator is inactive
4. Connect to pre-amplifier circuit via coaxial cables (RG-58, <30cm length)
5. Verify signal levels are within 20-80% of ADC range

### 7. Laplacian Operator Circuit

1. Install signal processing enclosure at position F1
2. Connect photodiode inputs using equal-length cables
3. Verify all ground connections share common reference point
4. Power circuit using linear power supply (not switching)
5. Test circuit operation with simulated inputs

### 8. 2D Chiral Signal Analyzer

1. Mount digital processing unit at position G1, outside main optical path
2. Connect to Laplacian output via shielded differential cables
3. Install FPGA configuration via secure USB connection
4. Verify self-test sequence completes successfully
5. Connect data acquisition computer via fiber optic link

## Vacuum System Setup (If Applicable)

1. Position vacuum chamber to enclose critical optical components
2. Install optical feedthroughs for beam paths
3. Connect vacuum pump through vibration isolator
4. Evacuate chamber to 10⁻⁶ torr and verify stable pressure

## System Validation

### Initial Testing

1. Power on all components in sequence according to the startup procedure
2. Verify all indicator lights show normal operation
3. Run system self-test sequence from control software
4. Validate open optical paths with alignment laser

### Calibration Procedure

1. Run the `calibration.py` script from the control computer
2. Follow on-screen instructions for each component calibration
3. Save calibration results to `calibration_data/` with timestamp
4. Verify all parameters are within tolerance limits

### Performance Verification

1. Execute the `test_sequence.py` script
2. Verify signal levels at all test points
3. Confirm chiral signal detection with test pattern
4. Document all measurements in the lab notebook

## Safety Considerations

### Laser Safety

- Wear appropriate laser safety goggles rated for the source wavelength
- Install laser interlock system connected to laboratory door
- Display appropriate laser warning signs at all entrances
- Never look directly into beam paths, even with protective eyewear

### Electrical Safety

- Ensure all equipment is properly grounded
- Use only UL/CE certified power supplies
- Keep all cables organized and away from walkways
- Implement emergency power shutoff accessible from multiple locations

### General Laboratory Safety

- Maintain clear access to all emergency exits
- Store all chemicals in appropriate cabinets when not in use
- Keep fire extinguisher and first aid kit easily accessible
- Review safety procedures with all laboratory personnel prior to experiments

## Troubleshooting Guide

### Common Issues

| Problem | Possible Cause | Solution |
|---------|---------------|----------|
| No laser output | Power supply disconnected | Check all power connections |
| | Interlock triggered | Verify door closure and reset interlock |
| Unstable readings | Vibration interference | Check vibration isolation system |
| | EMI | Verify all shielding is intact |
| Oscillator failure | Driver circuit issue | Check voltage levels on driver board |
| | Mechanical binding | Inspect for physical obstruction |
| | Grating damage | Examine grating surface under microscope |
| No chiral signal | Misalignment | Re-run alignment procedure |
| | Incorrect phase relationship | Adjust oscillator timing parameters |

### Diagnostic Tools

- Oscilloscope for signal verification
- Optical power meter for beam strength measurement
- Spectrum analyzer for frequency domain analysis
- Data logging system for long-term stability assessment

## Experimental Protocol

For detailed experimental protocols, refer to the `protocols/` directory, which contains specific procedures for different experimental configurations and research questions.

Always document any deviations from standard setup in the laboratory notebook, including environmental conditions, component substitutions, and calibration values.
