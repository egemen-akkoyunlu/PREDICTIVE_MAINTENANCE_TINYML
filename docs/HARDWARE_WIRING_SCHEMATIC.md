# Predictive Maintenance Node — Perfboard Wiring Schematic

> **Defense Industry Standard** | Rev 1.0 | February 2026

## 1. Bill of Materials (BOM)

| Qty | Component | Part Number / Notes |
|-----|-----------|---------------------|
| 1 | ESP32 DevKit V1 (30-pin) | MCU |
| 1 | MPU6050 Module | I2C IMU, Address 0x68 |
| 1 | INMP441 Module | I2S MEMS Microphone |
| 1 | MicroSD Card Module | SPI Interface (with onboard 3.3V regulator) |
| 1 | Red LED 3mm | Status Indicator |
| 1 | 220Ω Resistor | LED Current Limiting (10-15mA @ 3.3V) |
| 4 | 100nF (0.1µF) Ceramic Capacitor | Decoupling - C0G/X7R, 0805 |
| 2 | 10µF Electrolytic/Tantalum Capacitor | Bulk Decoupling |
| 1 | AMS1117-3.3 LDO (Optional) | External 3.3V Regulator if needed |

---

## 2. ESP32 DevKit V1 Pinout Strategy

### 2.1 Pin Selection Rationale

| Category | Avoid | Reason |
|----------|-------|--------|
| **Input-Only GPIOs** | GPIO34, GPIO35, GPIO36, GPIO39 | Cannot be used as outputs |
| **Strapping Pins** | GPIO0, GPIO2, GPIO12, GPIO15 | Affect boot mode |
| **Flash SPI Pins** | GPIO6-11 | Reserved for internal flash |
| **USB UART** | GPIO1 (TX), GPIO3 (RX) | Used for programming/debug |

### 2.2 Final Pin Assignment Table

| Function | GPIO | ESP32 Pin | Notes |
|----------|------|-----------|-------|
| **I2C (MPU6050)** ||||
| SDA | GPIO21 | D21 | Default I2C, internal pull-up OK |
| SCL | GPIO22 | D22 | Default I2C, internal pull-up OK |
| INT (Wake) | GPIO27 | D27 | RTC GPIO - Deep sleep capable |
| **I2S (INMP441)** ||||
| BCLK (SCK) | GPIO26 | D26 | Bit Clock |
| WS (LRCLK) | GPIO25 | D25 | Word Select |
| SD (DATA) | GPIO33 | D33 | Serial Data In |
| **SPI (SD Card)** ||||
| MOSI | GPIO23 | D23 | VSPI Default |
| MISO | GPIO19 | D19 | VSPI Default |
| SCK | GPIO18 | D18 | VSPI Default |
| CS | GPIO5 | D5 | Chip Select |
| **Indicator** ||||
| Status LED | GPIO4 | D4 | Via 220Ω resistor |
| **Power** ||||
| 3.3V OUT | 3V3 | 3.3V | Max 500mA total |
| GND | GND | GND | Common ground |
| VIN | VIN | 5V Input | USB or external 5V |

---

## 3. Complete Pin-to-Pin Netlist

### 3.1 MPU6050 Module (I2C)

```
┌─────────────────────────────────────────────────────────────────┐
│ MPU6050 Pin    │ Wire Color   │ ESP32 Pin   │ Notes            │
├─────────────────────────────────────────────────────────────────┤
│ VCC            │ RED          │ 3.3V        │ Via 100nF + 10µF │
│ GND            │ BLACK        │ GND         │ Common ground    │
│ SDA            │ BLUE         │ GPIO21      │ I2C Data         │
│ SCL            │ YELLOW       │ GPIO22      │ I2C Clock        │
│ INT            │ GREEN        │ GPIO27      │ Motion interrupt │
│ AD0            │ BLACK        │ GND         │ Address = 0x68   │
│ XDA            │ —            │ NC          │ Not connected    │
│ XCL            │ —            │ NC          │ Not connected    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 INMP441 Module (I2S)

```
┌─────────────────────────────────────────────────────────────────┐
│ INMP441 Pin    │ Wire Color   │ ESP32 Pin   │ Notes            │
├─────────────────────────────────────────────────────────────────┤
│ VDD            │ RED          │ 3.3V        │ Via 100nF + 10µF │
│ GND            │ BLACK        │ GND         │ Common ground    │
│ SD             │ WHITE        │ GPIO33      │ Serial Data Out  │
│ SCK            │ ORANGE       │ GPIO26      │ Bit Clock        │
│ WS             │ PURPLE       │ GPIO25      │ Word Select      │
│ L/R            │ BLACK        │ GND         │ Left channel     │
└─────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> L/R pin determines stereo channel: **GND = Left**, **VDD = Right**

### 3.3 MicroSD Card Module (SPI)

```
┌─────────────────────────────────────────────────────────────────┐
│ SD Module Pin  │ Wire Color   │ ESP32 Pin   │ Notes            │
├─────────────────────────────────────────────────────────────────┤
│ VCC / 5V       │ RED          │ VIN (5V)    │ Module has LDO   │
│ GND            │ BLACK        │ GND         │ Common ground    │
│ MOSI           │ BLUE         │ GPIO23      │ Master Out       │
│ MISO           │ GREEN        │ GPIO19      │ Master In        │
│ SCK            │ YELLOW       │ GPIO18      │ SPI Clock        │
│ CS             │ ORANGE       │ GPIO5       │ Chip Select      │
└─────────────────────────────────────────────────────────────────┘
```

> [!NOTE]
> Most SD card modules have onboard 3.3V regulator. Power from 5V (VIN) for stability.

### 3.4 Status LED

```
┌─────────────────────────────────────────────────────────────────┐
│ Component      │ Connection                                    │
├─────────────────────────────────────────────────────────────────┤
│ GPIO4          │ ──────[220Ω]──────┤>──────── GND              │
│                │                   LED                         │
│                │                (Anode)  (Cathode)             │
└─────────────────────────────────────────────────────────────────┘

Current = (3.3V - 1.8V) / 220Ω ≈ 6.8mA (safe, visible)
```

---

## 4. Power Distribution Schematic

### 4.1 Power Budget Analysis

| Component | Typical Current | Peak Current |
|-----------|-----------------|--------------|
| ESP32 (Active) | 80 mA | 240 mA (WiFi TX) |
| MPU6050 (Cycle Mode) | 0.5 mA | 3.5 mA |
| INMP441 (Active) | 1.4 mA | 2.5 mA |
| SD Card (Write) | 50 mA | 100 mA |
| LED | 7 mA | 7 mA |
| **Total** | **~140 mA** | **~350 mA** |

> [!WARNING]
> ESP32 DevKit's onboard AMS1117 can supply **max 800mA** at 3.3V, but thermal limits apply. For reliable operation with SD card writes, stay under **500mA**.

### 4.2 Power Distribution Topology

```
                                    ┌─────────────────────────┐
        USB 5V ──┬─────────────────►│ VIN (ESP32 DevKit)     │
                 │                  │                         │
                 │                  │ Onboard AMS1117-3.3    │
                 │                  │         ↓               │
                 │                  │ 3.3V Pin ───────────────┼──┬──► MPU6050 VCC
                 │                  │                         │  │    + C1 (100nF)
                 │                  │                         │  │    + C2 (10µF)
                 │                  │                         │  │
                 │                  │                         │  ├──► INMP441 VDD
                 │                  │                         │  │    + C3 (100nF)
                 │                  │                         │  │    + C4 (10µF)
                 │                  │                         │  │
                 │                  │                         │  └──► LED (via R1)
                 │                  │                         │
                 ├──────────────────┼─────────────────────────┼──► SD Module VCC (5V)
                 │                  │                         │
        GND ─────┴──────────────────┴─────────────────────────┴──► All GND (Star)
```

### 4.3 Optional: External LDO (Recommended for Production)

If you experience brownouts during SD card writes or WiFi, add:

```
USB 5V ──────[AMS1117-3.3]──────► 3.3V Rail (sensors only)
                │
               GND
               
Components for AMS1117:
- Input:  10µF electrolytic to GND
- Output: 10µF electrolytic + 100nF ceramic to GND
```

---

## 5. Decoupling Capacitor Placement

### 5.1 Capacitor Values and Locations

| Location | Capacitor | Type | Purpose |
|----------|-----------|------|---------|
| MPU6050 VCC-GND | 100nF | Ceramic C0G/X7R | High-frequency noise |
| MPU6050 VCC-GND | 10µF | Electrolytic | Bulk decoupling |
| INMP441 VDD-GND | 100nF | Ceramic C0G/X7R | **CRITICAL** for audio quality |
| INMP441 VDD-GND | 10µF | Electrolytic | Power stability |
| SD Card VCC-GND | 100nF | Ceramic | SPI noise suppression |
| ESP32 3.3V-GND | 100nF | Ceramic | MCU decoupling |

### 5.2 INMP441 Critical Placement

```
  ┌──────────────────────────────────────────────────────────────┐
  │              INMP441 Module (Top View)                       │
  │                                                              │
  │         [VDD] [GND] [SD] [WS] [SCK] [L/R]                   │
  │           │     │                                            │
  │           │     │     ← Place caps HERE (<5mm from pins)    │
  │          ═╧═══════╧═                                         │
  │          C3(100nF) + C4(10µF)                                │
  │                                                              │
  │  Maximum distance from VDD/GND pins: 5mm                     │
  │  Use short, wide traces or direct wire                       │
  └──────────────────────────────────────────────────────────────┘
```

> [!CAUTION]
> **INMP441 is extremely sensitive to power supply noise!**
> Poor decoupling causes audible artifacts and false anomaly detections.

---

## 6. Perfboard Layout Advice

### 6.1 Signal Routing Hierarchy

```
Priority 1 (Most Critical):  I2S Signals (BCLK, WS, SD)
Priority 2:                  SPI Signals (SCK, MOSI, MISO)
Priority 3:                  I2C Signals (SDA, SCL)
Priority 4:                  GPIO/Interrupt lines
```

### 6.2 Recommended Physical Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PERFBOARD TOP VIEW                          │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                             │   │
│   │    ╔═══════════════════════════════════════════════════╗    │   │
│   │    ║           ESP32 DEVKIT V1 (30-PIN)               ║    │   │
│   │    ║   [EN]  [VP] [VN] [D34]...[D21][D22][TX][RX]     ║    │   │
│   │    ║                     ┃                             ║    │   │
│   │    ║   [VIN][GND][D13]...[D26][D25][D33][D32]         ║    │   │
│   │    ╚═══════════════════════════════════════════════════╝    │   │
│   │                          │                                  │   │
│   │     ┌────────┐          │          ┌────────┐              │   │
│   │     │MPU6050 │◄─────────┘          │INMP441 │              │   │
│   │     │(I2C)   │   Keep              │(I2S)   │              │   │
│   │     │        │   short!            │        │              │   │
│   │     └────────┘                     └────────┘              │   │
│   │         │                              │                    │   │
│   │        [C]                            [C]                   │   │
│   │     Decoupling                     Decoupling               │   │
│   │                                                             │   │
│   │              ┌──────────────┐                               │   │
│   │              │  SD CARD     │  ← Away from I2S!            │   │
│   │              │  MODULE      │                               │   │
│   │              └──────────────┘                               │   │
│   │                                                             │   │
│   │     [LED] ─[R]─ GPIO4                Ground Plane           │   │
│   │                                      (copper pour)          │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.3 Critical Layout Rules

| Rule | Description |
|------|-------------|
| **1. Star Ground** | All GND connections meet at single point near ESP32 |
| **2. I2S Separation** | Keep INMP441 signals ≥15mm from SPI/power traces |
| **3. Wire Length** | I2S wires < 50mm, I2C wires < 100mm |
| **4. Parallel Avoidance** | Never run I2S and SPI clocks parallel for >10mm |
| **5. Ground Guard** | Run GND wire between I2S and SPI traces |
| **6. Twist Critical Pairs** | Twist SCK+GND and SD+GND for INMP441 |

### 6.4 Soldering Sequence

1. **Mount the ESP32 headers first** — align and tack corners
2. **Install power rails** — 3.3V and GND bus wires
3. **Place all decoupling capacitors** — before modules!
4. **Mount MPU6050** — I2C is most tolerant
5. **Mount SD Card Module** — SPI next
6. **Mount INMP441 last** — most sensitive, shortest wires possible
7. **Add LED circuit** — lowest priority
8. **Verify all connections** with multimeter before powering

---

## 7. Signal Interference Mitigation

### 7.1 Clock Frequency Analysis

| Signal | Frequency | Wavelength | Critical Length |
|--------|-----------|------------|-----------------|
| I2C SCL | 400 kHz | 750m | Not critical |
| I2S BCLK | 1.024 MHz | 293m | >50mm matters |
| SPI SCK | 4-20 MHz | 15-75m | Keep under 30mm |

### 7.2 Crosstalk Prevention Matrix

```
        │ I2C  │ I2S  │ SPI  │ 
────────┼──────┼──────┼──────┤
 I2C    │  —   │ LOW  │ LOW  │
 I2S    │ LOW  │  —   │ HIGH │  ← SPI and I2S conflict!
 SPI    │ LOW  │ HIGH │  —   │
```

**Solution**: Physical separation OR ground wire between I2S and SPI traces.

### 7.3 Wire Recommendations

| Signal Type | Wire Gauge | Recommendation |
|-------------|------------|----------------|
| Power (3.3V, GND) | 22-24 AWG | Solid core |
| I2C, GPIO | 26-28 AWG | Stranded OK |
| I2S | 26 AWG | **Shielded or twisted pair** |
| SPI | 24-26 AWG | Keep short |

---

## 8. Firmware Pin Configuration Update

Update your `anomaly_detector.cpp` to match this schematic:

```cpp
namespace config {
    // I2C Pins (MPU6050)
    constexpr gpio_num_t I2C_SDA_PIN = GPIO_NUM_21;
    constexpr gpio_num_t I2C_SCL_PIN = GPIO_NUM_22;
    constexpr gpio_num_t MPU_INT_PIN = GPIO_NUM_27;
    
    // I2S Pins (INMP441)
    constexpr gpio_num_t I2S_BCK_PIN  = GPIO_NUM_26;
    constexpr gpio_num_t I2S_WS_PIN   = GPIO_NUM_25;
    constexpr gpio_num_t I2S_DATA_PIN = GPIO_NUM_33;
    
    // SPI Pins (SD Card) - VSPI
    constexpr gpio_num_t SPI_MOSI_PIN = GPIO_NUM_23;
    constexpr gpio_num_t SPI_MISO_PIN = GPIO_NUM_19;
    constexpr gpio_num_t SPI_SCK_PIN  = GPIO_NUM_18;
    constexpr gpio_num_t SD_CS_PIN    = GPIO_NUM_5;
    
    // Indicator
    constexpr gpio_num_t LED_STATUS_PIN = GPIO_NUM_4;
}
```

---

## 9. Pre-Power Checklist

Before applying power, verify with multimeter:

- [ ] 3.3V rail shows NO short to GND (resistance > 1kΩ)
- [ ] All GND connections have continuity
- [ ] I2C SDA/SCL not shorted together
- [ ] SPI MOSI/MISO not shorted together
- [ ] LED polarity correct (flat side = cathode = GND)
- [ ] All capacitors oriented correctly (electrolytic only)
- [ ] No solder bridges between adjacent pins

---

## 10. Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│              PREDICTIVE MAINTENANCE NODE - QUICK REF            │
├─────────────────────────────────────────────────────────────────┤
│  MPU6050:  SDA=21, SCL=22, INT=27, ADDR=0x68                   │
│  INMP441:  BCLK=26, WS=25, SD=33, L/R=GND                      │
│  SD CARD:  MOSI=23, MISO=19, SCK=18, CS=5                      │
│  LED:      GPIO4 → 220Ω → LED → GND                            │
│  POWER:    USB 5V → VIN, Sensors from 3.3V pin                 │
├─────────────────────────────────────────────────────────────────┤
│  DECOUPLING: 100nF + 10µF at each sensor VCC                   │
│  CRITICAL:   INMP441 caps < 5mm from pins!                     │
└─────────────────────────────────────────────────────────────────┘
```

---

*Document generated for defense-grade reliability. Verify all connections before deployment.*
