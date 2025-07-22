// #include <stdbool.h>
// #include <stdint.h>
// #include "deck.h"
// #include "i2cdev.h"
// #include "debug.h"
// #include "FreeRTOS.h"
// #include "task.h"

// #define I2C_BUS I2C1_DEV

// static void i2cScanInit(struct deckInfo_s *deck)
// {

//   DEBUG_PRINT("⚙️ Starting I2C Bus Scan...\n");
//    i2cdevInit(I2C1_DEV);

//   vTaskDelay(M2T(300)); // Wait 100ms for sensors to power on

//   uint8_t found_count = 0;
//   for (uint8_t addr = 0x03; addr <= 0x77; addr++)
//   {
//     // We attempt a 1-byte read. We don't care about the data,
//     // only whether the function returns 'true', which means a
//     // device at 'addr' acknowledged the communication.
//     uint8_t frame[6];
//     bool success = i2cdevRead(I2C_BUS, addr, 6, frame);
    
//     if (success) {
//       DEBUG_PRINT("✅ Device found at address 0x%02X\n", addr);
//       found_count++;
//     }
//   }

//   if (found_count == 0) {
//       DEBUG_PRINT("❌ No external I2C devices found. Check wiring!\n");
//   } else {
//       DEBUG_PRINT("Scan complete. Found %d device(s).\n", found_count);
//   }
// }

// static const DeckDriver i2c1scan = {
//   .name = "i2c1scan",
//   .init = i2cScanInit,
// };

// DECK_DRIVER(i2c1scan);
// THIS WORKS ^^

#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include "debug.h"
#include "deck.h"
#include "uart2.h"
#include "log.h"
#include "param.h"
#include "FreeRTOS.h"
#include "task.h"
#include "i2cdev.h"
#include "system.h"

// I2C and sensor configuration
#define I2C_BUS         I2C1_DEV
#define ADDRESS         0x35
#define CONFIG_REG_ADDR 0x10
#define CONFIG_BYTES    2
#define BEGIN_REG       0x20  // Measurement trigger value

// Sign-extend 12-bit value to 16-bit signed integer
#define SIGN_EXT_12BIT(val) (((val) & 0x800) ? ((val) | 0xF000) : (val))

// Global magnetic field readings
static int16_t bx = 0;
static int16_t by = 0;
static int16_t bz = 0;
static bool isInit = false;

/**
 * TLE493D sensors in Master-Controlled Mode (MOD1) do not respond to 1-byte I2C reads.
 * They require a 6- or 7-byte read from address 0x00 to acknowledge I2C.
 */

// Continuous sensor polling task
static void i2cReaderTask(void *param) {
    static uint8_t buf[7];
    systemWaitStart();

    while (1) {
        // Trigger new measurement
        uint8_t startTrig = BEGIN_REG;
        

        vTaskDelay(M2T(10)); // Wait for measurement to complete (≥3 ms)

        // Read sensor data (must read ≥6 bytes to get valid response)
        bool readSuccess = i2cdevRead(I2C_BUS, ADDRESS, 7, buf);

        if (readSuccess) {
            DEBUG_PRINT("Raw: %02X %02X %02X %02X %02X %02X %02X\n",
                        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6]);

            // Parse 12-bit signed values from sensor frame
            bx = SIGN_EXT_12BIT((buf[0] << 4) | ((buf[4] & 0xF0) >> 4));
            by = SIGN_EXT_12BIT((buf[1] << 4) | (buf[4] & 0x0F));
            bz = SIGN_EXT_12BIT((buf[2] << 4) | ((buf[5] & 0xF0) >> 4));
            DEBUG_PRINT("Bx: %d, By: %d, Bz: %d\n", bx, by, bz);

            i2cdevWriteReg8(I2C_BUS, ADDRESS, 0x00, 1, &startTrig);
            
            //i2cdevWriteByte(I2C_BUS, ADDRESS, I2CDEV_NO_MEM_ADDR, startTrig);
        } else {
            DEBUG_PRINT("❌ I2C read failed\n");
        }

        //vTaskDelay(M2T(10)); // Loop at ~100 Hz
        
        
    }
}

// Deck initialization
static void deck_i2c_loggerInit(struct deckInfo_s *info) {
    if (isInit) return;

    DEBUG_PRINT("🔧 Initializing TLE493D-W2B6 deck...\n");

    i2cdevInit(I2C_BUS);
    vTaskDelay(M2T(3));  // Sensor startup delay
    uint8_t frame[6];
    bool success = i2cdevRead(I2C_BUS, ADDRESS, 7, frame);
    
    if (success) {
      DEBUG_PRINT("✅ Device found at address 0x%02X\n", ADDRESS);
    }
    // Optional reset write to 0x00 (as in Arduino example)
    uint8_t resetByte = 0x00;
    i2cdevWriteReg8(I2C_BUS, 0x00, 0x00, 1, &resetByte);

    // Write CONFIG + MOD1 (at registers 0x10 and 0x11)
    uint8_t configData[2] = {
        0b00000000, // CONFIG: enable all axes, full range
        0b00011001  // MOD1: master-controlled mode, one-byte read, interrupt enable
    };
    i2cdevWriteReg8(I2C_BUS, ADDRESS, CONFIG_REG_ADDR, CONFIG_BYTES, configData);
    vTaskDelay(M2T(3));

    // Trigger initial measurement (0x20 to 0x00)
    uint8_t trig = BEGIN_REG;
    if (!i2cdevWriteReg8(I2C_BUS, ADDRESS, 0x00, 1, &trig)) {
        DEBUG_PRINT("❌ Initial trigger failed\n");
    } else {
        DEBUG_PRINT("✅ Initial trigger sent\n");
    }

    

    // Start polling task
    xTaskCreate(i2cReaderTask, "i2cReader", 512, NULL, 3, NULL);
    isInit = true;
}

// Deck test function
static bool helloI2C(void) {
    DEBUG_PRINT("TLE493D deck active!\n");
    return true;
}

// Deck registration macro
const DeckDriver i2c1scan = {
    .name = "i2c1scan",
    .init = deck_i2c_loggerInit,
    .test = helloI2C,
};
DECK_DRIVER(i2c1scan);

// // Logging for cfclient
// LOG_GROUP_START(i2c_logger)
// LOG_ADD(LOG_INT16, bx, &bx)
// LOG_ADD(LOG_INT16, by, &by)
// LOG_ADD(LOG_INT16, bz, &bz)
// LOG_GROUP_STOP(i2c_logger)
