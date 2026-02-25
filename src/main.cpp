#include <Arduino.h>
#include "driver/i2s.h"

#define I2S_MCLK_PIN 16
#define I2S_BCK_PIN 17
#define I2S_LRCK_PIN 18
#define I2S_DIN_PIN 8

#define SAMPLE_RATE 44100
#define TOTAL_SAMPLES 44032

int32_t recordingBuffer[TOTAL_SAMPLES];

void setup_i2s()
{
  i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT, // tell driver everything is 32-bit
      .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 256,
      .use_apll = true,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0,
      .mclk_multiple = I2S_MCLK_MULTIPLE_256,
      .bits_per_chan = I2S_BITS_PER_CHAN_DEFAULT, // same as bits_per_sample
  };

  i2s_pin_config_t pin_config = {
      .mck_io_num = I2S_MCLK_PIN,
      .bck_io_num = I2S_BCK_PIN,
      .ws_io_num = I2S_LRCK_PIN,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = I2S_DIN_PIN,
  };

  ESP_ERROR_CHECK(i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL));
  ESP_ERROR_CHECK(i2s_set_pin(I2S_NUM_0, &pin_config));
}

void setup()
{
  Serial.begin(115200);

  setup_i2s();
  // Serial.println("Driver up. Waiting 5 seconds — probe pins now...");
  delay(5000); // probe BCK/LRCK/MCLK during this window

  // Serial.println("Recording...");
  const int CHUNK_FRAMES = 128;
  int32_t readBuf[CHUNK_FRAMES * 2];
  size_t bytes_read = 0;
  int samplesCollected = 0;

  while (samplesCollected < TOTAL_SAMPLES)
  {
    int framesWanted = min(CHUNK_FRAMES, TOTAL_SAMPLES - samplesCollected);
    i2s_read(I2S_NUM_0, readBuf, framesWanted * 2 * sizeof(int32_t), &bytes_read, portMAX_DELAY);
    int framesRead = bytes_read / (2 * sizeof(int32_t));
    for (int i = 0; i < framesRead && samplesCollected < TOTAL_SAMPLES; i++)
    {
      // Channel order is RIGHT, LEFT
      int32_t raw = readBuf[i * 2 + 1]; // use LEFT channel instead

      // Convert 24-bit left-justified to proper signed 32-bit
      int32_t sample = raw >> 8; // arithmetic shift keeps sign

      recordingBuffer[samplesCollected++] = sample;
    }
  }

  // driver stays up after recording so clocks remain for PCM1808
  // Serial.println("Done. Printing samples...");
  for (int i = 0; i < TOTAL_SAMPLES; i++)
  {
    Serial.print(recordingBuffer[i]);
    Serial.print(',');
  }
  Serial.println();
}

void loop() {}