/**
 * TLI493D-W2BW Shield2Go 3D Magnetic Sensor Driver for Crazyflie 2.1
 * 
 * Reads magnetic field values (Bx, By, Bz) from Infineon TLI493D-W2BW sensor
 * via I2C and exposes them through the Crazyflie logging framework.
 * 
 * IMPORTANT: This is for TLI493D-W2BW which uses direct byte protocol (NO register addresses)
 * If you have TLE493D-A0, the protocol is different!
 * 
 * Hardware connections:
 * - SDA to Crazyflie SDA
 * - SCL to Crazyflie SCL
 * - VCC to 3.3V
 * - GND to GND
 * 
 * Pull-up resistors (2.2kΩ - 10kΩ) required on SDA/SCL if using bare sensor.
 * Shield2Go board has onboard pull-ups.
 */

#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "debug.h"
#include "deck.h"
#include "log.h"
#include "param.h"
#include "FreeRTOS.h"
#include "task.h"
#include "i2cdev.h"
#include "system.h"

// ============================================================================
// Configuration
// ============================================================================

#define I2C_BUS             I2C1_DEV
#define TLI493D_ADDRESS     0x35        // Default I2C address for Shield2Go
#define RESET_ADDRESS       0x00        // Special address for reset
#define DATA_READ_BYTES     7           // Bytes to read for sensor data
#define SENSOR_POLL_MS      10         // Polling interval (10 Hz)

// ============================================================================
// Global Variables
// ============================================================================

static int16_t bx = 0;  // Magnetic field X-axis (raw 12-bit value)
static int16_t by = 0;  // Magnetic field Y-axis (raw 12-bit value)
static int16_t bz = 0;  // Magnetic field Z-axis (raw 12-bit value)
static uint8_t temp = 0; // Temperature (raw 8-bit value)
static bool isInit = false;
static bool sensorError = false;

// ============================================================================
// I2C Helper Functions for Direct Byte Protocol
// ============================================================================

/**
 * Write bytes directly to sensor (no register address)
 * This matches Arduino's Wire.beginTransmission() + Wire.write() + Wire.endTransmission()
 */
static bool tli493d_writeBytes(uint8_t *data, uint8_t len) {
    return i2cdevWrite(I2C_BUS, TLI493D_ADDRESS, len, data);
}

/**
 * Read bytes directly from sensor (no register address)
 * This matches Arduino's Wire.requestFrom()
 */
static bool tli493d_readBytes(uint8_t *data, uint8_t len) {
    return i2cdevRead(I2C_BUS, TLI493D_ADDRESS, len, data);
}

// ============================================================================
// Sensor Reading Task
// ============================================================================

/**
 * Continuously polls the TLI493D sensor and updates global variables.
 * Protocol: Read 7 bytes, parse data, trigger next measurement
 */
static void tli493dReaderTask(void *param) {
    static uint8_t buf[DATA_READ_BYTES];
    uint32_t successCount = 0;
    uint32_t failCount = 0;
    
    systemWaitStart();
    
    DEBUG_PRINT("🔄 TLI493D polling task started\n");

    while (1) {
        // Read sensor data frame (7 bytes)
        bool readSuccess = tli493d_readBytes(buf, DATA_READ_BYTES);

        if (readSuccess) {
            // Debug: Print raw bytes periodically
            if (successCount % 10 == 0) {
                DEBUG_PRINT("Raw: %02X %02X %02X %02X %02X %02X %02X\n",
                           buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6]);
            }
            
            // Parse 12-bit signed magnetic field values (Arduino method)
            // Data format:
            // Byte 0: Bx[11:4]
            // Byte 1: By[11:4]  
            // Byte 2: Bz[11:4]
            // Byte 3: Temp[11:4]
            // Byte 4: Bx[3:0] (upper nibble), By[3:0] (lower nibble)
            // Byte 5: Bz[3:0] (upper nibble), Temp[3:0] (lower nibble)
            
            bx = (int16_t)((buf[0] << 8) | (buf[4] & 0xF0)) >> 4;
            by = (int16_t)((buf[1] << 8) | ((buf[4] & 0x0F) << 4)) >> 4;
            bz = (int16_t)((buf[2] << 8) | ((buf[5] & 0x0F) << 4)) >> 4;
            temp = buf[3];
            
            successCount++;
            sensorError = false;
            
            // Periodic debug output
            if (successCount % 10 == 0) {
                DEBUG_PRINT("TLI493D [%lu]: Bx=%d, By=%d, Bz=%d, T=%d\n", 
                           successCount, bx, by, bz, temp);
            }
            
            // Trigger next measurement: Write 0x20 directly to sensor
            uint8_t triggerByte = 0x20;  // 0b00100000
            tli493d_writeBytes(&triggerByte, 1);
            
        } else {
            failCount++;
            sensorError = true;
            
            if (failCount % 10 == 0) {
                DEBUG_PRINT("❌ TLI493D I2C read failed (%lu failures)\n", failCount);
            }
        }

        vTaskDelay(M2T(SENSOR_POLL_MS));
    }
}

// ============================================================================
// Deck Initialization
// ============================================================================

/**
 * Initializes the TLI493D sensor in Master-Controlled Mode.
 * Protocol matches Arduino setup() function.
 */
static void tli493dDeckInit(struct deckInfo_s *info) {
    if (isInit) {
        DEBUG_PRINT("⚠️  TLI493D already initialized\n");
        return;
    }

    DEBUG_PRINT("🔧 Initializing TLI493D-W2BW Shield2Go...\n");

    // Initialize I2C bus
    i2cdevInit(I2C_BUS);
    vTaskDelay(M2T(3));
    
    // Set I2C clock to 400kHz (matching Arduino)
    // Note: i2cdevInit may already set this, but ensuring it's correct
    
    // Reset sensor (Arduino: Wire.beginTransmission(0x00); Wire.write(0x00);)
    DEBUG_PRINT("🔄 Resetting sensor...\n");
    uint8_t resetByte = 0x00;
    if (!i2cdevWrite(I2C_BUS, RESET_ADDRESS, 1, &resetByte)) {
        DEBUG_PRINT("⚠️  Reset command may have failed (this is sometimes OK)\n");
    }
    vTaskDelay(M2T(3));

    // Test if sensor responds
    uint8_t testBuf[7];
    if (!tli493d_readBytes(testBuf, 7)) {
        DEBUG_PRINT("❌ TLI493D not found at 0x%02X - check wiring!\n", TLI493D_ADDRESS);
        DEBUG_PRINT("   Verify: SDA, SCL, VCC(3.3V), GND connections\n");
        DEBUG_PRINT("   Pull-ups: Need 2.2k-10k on SDA/SCL if using bare sensor\n");
        return;
    }
    DEBUG_PRINT("✅ TLI493D found at address 0x%02X\n", TLI493D_ADDRESS);

    // Configure sensor (Arduino: writes to register 0x10)
    // Write CONFIG (0x00) and MOD1 (0x19) bytes
    // Note: For TLI493D-W2BW we write 3 bytes: reg_addr, config, mod1
    DEBUG_PRINT("⚙️  Configuring sensor...\n");
    uint8_t configBytes[3] = {
        0x10,        // Register address
        0b00000000,  // CONFIG: Enable all axes, full range
        0b00011001   // MOD1: INT_EN=1, Fast=1, MCM=1
    };
    
    if (!tli493d_writeBytes(configBytes, 3)) {
        DEBUG_PRINT("❌ Configuration write failed\n");
        return;
    }
    DEBUG_PRINT("✅ Configuration written\n");
    vTaskDelay(M2T(3));

    // Trigger initial measurement
    uint8_t triggerByte = 0x20;
    if (!tli493d_writeBytes(&triggerByte, 1)) {
        DEBUG_PRINT("⚠️  Initial trigger failed\n");
    } else {
        DEBUG_PRINT("✅ Initial measurement triggered\n");
    }
    vTaskDelay(M2T(10));

    // Verify sensor responds with valid data
    uint8_t testRead[DATA_READ_BYTES];
    if (tli493d_readBytes(testRead, DATA_READ_BYTES)) {
        int16_t test_bx = (int16_t)((testRead[0] << 8) | (testRead[4] & 0xF0)) >> 4;
        int16_t test_by = (int16_t)((testRead[1] << 8) | ((testRead[4] & 0x0F) << 4)) >> 4;
        int16_t test_bz = (int16_t)((testRead[2] << 8) | ((testRead[5] & 0x0F) << 4)) >> 4;
        DEBUG_PRINT("✅ Initial reading: Bx=%d, By=%d, Bz=%d\n", test_bx, test_by, test_bz);
        DEBUG_PRINT("   Raw bytes: %02X %02X %02X %02X %02X %02X %02X\n",
                   testRead[0], testRead[1], testRead[2], testRead[3], 
                   testRead[4], testRead[5], testRead[6]);
    }

    // Start background polling task
    xTaskCreate(tli493dReaderTask, "TLI493D", 512, NULL, 3, NULL);
    
    isInit = true;
    DEBUG_PRINT("🚀 TLI493D initialization complete!\n");
    DEBUG_PRINT("   Access data via cfclient: tli493d.bx, .by, .bz\n");
}

// ============================================================================
// Deck Test Function
// ============================================================================

static bool tli493dDeckTest(void) {
    if (!isInit) {
        DEBUG_PRINT("❌ TLI493D not initialized\n");
        return false;
    }
    
    if (sensorError) {
        DEBUG_PRINT("⚠️  TLI493D has communication errors\n");
        return false;
    }
    
    DEBUG_PRINT("✅ TLI493D deck test passed (Bx=%d, By=%d, Bz=%d)\n", bx, by, bz);
    return true;
}

// ============================================================================
// Deck Driver Registration
// ============================================================================

static const DeckDriver tli493d_deck = {
    .name = "nomcuWindSensor",
    .init = tli493dDeckInit,
    .test = tli493dDeckTest,
};

DECK_DRIVER(tli493d_deck);

// ============================================================================
// Crazyflie Logging Framework
// ============================================================================

/**
 * Exposes sensor data to cfclient via logging framework.
 * Access in cfclient with: tli493d.bx, tli493d.by, tli493d.bz, tli493d.temp
 */
LOG_GROUP_START(tli493d)
LOG_ADD(LOG_INT16, bx, &bx)
LOG_ADD(LOG_INT16, by, &by)
LOG_ADD(LOG_INT16, bz, &bz)
LOG_ADD(LOG_UINT8, temp, &temp)
LOG_ADD(LOG_UINT8, error, &sensorError)
LOG_GROUP_STOP(tli493d)

// ============================================================================
// Parameters
// ============================================================================

PARAM_GROUP_START(tli493d)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, isInit, &isInit)
PARAM_GROUP_STOP(tli493d)
