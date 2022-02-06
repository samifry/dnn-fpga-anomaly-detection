/*
 	 **HOST LINUX APPLICATION**

 	 Master Thesis - Model-Based Predictive Maintenance Tool on FPGA
 	 Master: Mechatronics Engineering
 	 Author: Sami Foery
 	 Date : 26.01.2022

 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>
#include <fcntl.h>
#include <memory.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <linux/ioctl.h>
#include <linux/spi/spidev.h>
#include <linux/types.h>

// If not define -> TRAINING MODE (only data acquisition)
#define INFERENCE_MODE

// SPI SLAVE DEVICE REGISTERS
#define ARRAY_SIZE(a) 				(sizeof(a) / sizeof((a)[0]))
#define READREG(a) 					(a | 0x80) // macro to read register in SPI slave
#define WHO_AM_I_REG				0x75
#define PM_REG 						0x6B
#define SENSI_REG 					0x1C
#define ACCEL_XREG1 				0x3B
#define ACCEL_XREG2 				0x3C

// AXI GPIO REGISTERS
#define GPIO_DATA 					0x0000
#define GPIO_TRI 					0x0004

// SOURCE REGISTER TO SAVE DATA
#define SOURCE_ADDRESS 				0x10000000

// LIMIT COMPARISON FOR NEGATIVE VALUES
#define LIM							50000

// LENGTH MEMORY MAP AND FRACTIONAL PRECISION
#define LENGTH 						getpagesize()*16
#define MAXBUFF 					LENGTH/4
#define FIXED_POINT_FRACTIONAL_BITS 8

// REFERENCES FROM TRAIN DATA
#define TRAIN_MAX					0.248779
#define TRAIN_MIN					-0.197998

// SPI INITIALIZATION VALUES
static uint32_t mode;
static uint8_t bits = 8;
static uint32_t speed = 50000;
static uint16_t delay = 0;

static void pabort(const char *s)
{
	perror(s);
	abort();
}


// SPI transfer function
static void transfer(int fd, uint8_t const *tx, uint8_t const *rx, size_t len)
{
	int ret;
	struct spi_ioc_transfer tr = {
		.tx_buf = (unsigned long)tx,
		.rx_buf = (unsigned long)rx,
		.len = len,
		.delay_usecs = delay,
		.speed_hz = speed,
		.bits_per_word = bits,
	};

	ret = ioctl(fd, SPI_IOC_MESSAGE(1), &tr);
	if (ret < 1)
		pabort("can't send spi message");
}

// SPI data acquisition
void* spi_data(float src[MAXBUFF])
{

	uint8_t comm_config_tx[] = {READREG(WHO_AM_I_REG), 0xFF};
	uint8_t pm_config_tx[] = {PM_REG, 0x01};
	uint8_t sensi_config_tx[] = {SENSI_REG, 0x00};

	uint8_t accel_tx[] = {
		READREG(ACCEL_XREG1), READREG(ACCEL_XREG2), 0xFF,
	};

	uint8_t comm_config_rx[ARRAY_SIZE(comm_config_tx)] = {0, };
	uint8_t pm_config_rx[ARRAY_SIZE(pm_config_tx)] = {0, };
	uint8_t sensi_config_rx[ARRAY_SIZE(sensi_config_tx)] = {0, };
	uint8_t accel_rx[ARRAY_SIZE(accel_tx)] = {0, };


	static const char *device = "/dev/spidev1.0";
	int fd;
	int ret = 0;
	int i = 0;

	fd = open(device, O_RDWR);
	if (fd <= 0) {
		perror("Unable to open the /dev/spidev1.0 driver\n");
		exit(1);
	};

	/*
	 * Spi mode
	 */
	ret = ioctl(fd, SPI_IOC_WR_MODE, &mode);
	if (ret == -1)
		pabort("can't set spi mode");

	ret = ioctl(fd, SPI_IOC_RD_MODE, &mode);
	if (ret == -1)
		pabort("can't get spi mode");

	/*
	 * Bits per word
	 */
	ret = ioctl(fd, SPI_IOC_WR_BITS_PER_WORD, &bits);
	if (ret == -1)
		pabort("can't set bits per word");

	ret = ioctl(fd, SPI_IOC_RD_BITS_PER_WORD, &bits);
	if (ret == -1)
		pabort("can't get bits per word");

	/*
	 * Max speed Hz
	 */
	ret = ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed);
	if (ret == -1)
		pabort("can't set max speed hz");

	ret = ioctl(fd, SPI_IOC_RD_MAX_SPEED_HZ, &speed);
	if (ret == -1)
		pabort("can't get max speed hz");

	printf("spi mode: 0x%x\n", mode);
	printf("bits per word: %d\n", bits);
	printf("max speed: %d Hz (%d KHz)\n", speed, speed/1000);

	// Communication test with slave SPI device using "WHO_AM_I" register
	transfer(fd, comm_config_tx, comm_config_rx, sizeof(comm_config_tx)); // Control communication with slave device


	if (comm_config_rx[1] == 0x68){
		printf("Communication with slave SPI device successful\n");
	}
	else{
		printf("Communication with slave SPI device failed\n");
		exit(1);
	}


	// Power management configuration
	transfer(fd, pm_config_tx, pm_config_rx, sizeof(pm_config_tx));
	printf("Power management successfully configured\n");

	// Sensibility configuration to +/- 2g (1.12 mg/LSB)
	transfer(fd, sensi_config_tx, sensi_config_rx, sizeof(sensi_config_tx));
	printf("Sensibility successfully configured to +/- 2g\n");

	// Acceleration X data acquisition
	printf("Saving acceleration X axis data in the main memory . . .\n");
	for (i=0; i<MAXBUFF; i++){

		transfer(fd, accel_tx, accel_rx, sizeof(accel_tx));

		usleep(1000);

		src[i] = ((accel_rx[1] << 8) | accel_rx[2]);

		if (src[i] > LIM){
			src[i] = (src[i] - LENGTH)/16384.0f;
		}
		else{
			src[i] = src[i]/16384.0f;
		}
	}
	printf("X axis acceleration data successfully saved in memory\n");

	close(fd);
}

// Function for loading offline data file to the memory
float* load_data(char* sdmem, float src[MAXBUFF], float save[MAXBUFF])
{
	int i=0;
	char row[MAXBUFF];

	FILE *fp = fopen(sdmem, "r");

	if (fp == NULL) {
		perror("Unable to open the file\n");
		exit(1);
	};

	printf("Loading and saving data to memory . . .\n");

	for (i=0; i<MAXBUFF; i++)
	{
		fgets(row, MAXBUFF, fp);
		src[i] = atof(row);
		save[i] = atof(row);
		i = i+1;
	};
	return src;
}

// Function for the normalization of the data using training data based
void* normalize_data(float src[MAXBUFF], float save[MAXBUFF])
{
	int i = 0;

	for (i=0; i<MAXBUFF; i++){
		src[i] = (src[i] - TRAIN_MIN) / (TRAIN_MAX - TRAIN_MIN);
		save[i] = src[i];
	}
}

// Floating point data to fixed point transformation
void* to_fixed_point(float* data, unsigned int src[MAXBUFF])
{
	    unsigned int i = 0;

	    printf("Floating point to fixed point transformation . . .\n");

	    for (i=0; i<MAXBUFF; i++){
	        src[i] = (int) (data[i] * (1 << FIXED_POINT_FRACTIONAL_BITS));
	    };
}

// Logs creation from saved data on source register
void* create_logs(float* logs, const char* name)
{
	char* path = "/run/media/mmcblk0p1/";
	unsigned int i = 0;

	FILE *fps;

    char * filename = (char *) malloc(1 + strlen(path)+ strlen(name) );
    strcpy(filename, path);
    strcat(filename, name);
	printf("\n Creating %s file\n",filename);

	fps = fopen(filename,"w+");
	if (fps == NULL) {
		perror("Unable to open the file\n");
		exit(1);
	};

	for(i=0;i<MAXBUFF;i++){

		fprintf(fps,"%f\n",logs[i]);

	}

	fclose(fps);

	printf("%s file created\n",filename);

}

int main(int argc, char *argv[])
{
	int i=0, fd0, fd1, fd2, fd3;
	char *uiod0 = "/dev/uio0";
	char *uiod1 = "/dev/uio1";
	char *uiod2 = "/dev/uio2";
	char *uiod3 = "/dev/uio3";
	char* sdmem = "/run/media/mmcblk0p1/full_train.dat";
	float* data, logs[MAXBUFF];
	float save_data[MAXBUFF];
	float RECSTR;
	void *gpio_ptr0;
	void *gpio_ptr1;
	void *gpio_ptr2;
	void *gpio_ptr3;

	struct timeval start, end;

    // Open the UIO device file to allow access to the device in user space

    fd0 = open(uiod0, O_RDWR);
    if (fd0 < 1) {
    	printf("Invalid UIO0 device file:%s.\n", uiod0);
    }

    fd1 = open(uiod1, O_RDWR);
	if (fd1 < 1) {
		printf("Invalid UIO1 device file:%s.\n", uiod1);
	}
	fd2 = open(uiod2, O_RDWR);
	if (fd2 < 1) {
		printf("Invalid UIO2 device file:%s.\n", uiod2);
	}
	fd3 = open(uiod3, O_RDWR);
	if (fd3 < 1) {
		printf("Invalid UIO3 device file:%s.\n", uiod3);
	}

	// Mmap the AXI GPIOs devices into user space

    gpio_ptr0 = mmap(NULL, getpagesize(), PROT_READ|PROT_WRITE, MAP_SHARED, fd0, 0);
    // If addr is NULL, then the kernel chooses the (page-aligned) address at which to create the mapping
    if (gpio_ptr0 == MAP_FAILED) {
    	printf("Mmap call failure.\n");
    	return -1;
    }
    gpio_ptr1 = mmap(NULL, getpagesize(), PROT_READ|PROT_WRITE, MAP_SHARED, fd1, 0);
    // If addr is NULL, then the kernel chooses the (page-aligned) address at which to create the mapping
    if (gpio_ptr1 == MAP_FAILED) {
    	printf("Mmap call failure.\n");
    	return -1;
    }
    gpio_ptr2 = mmap(NULL, getpagesize(), PROT_READ|PROT_WRITE, MAP_SHARED, fd2, 0);
    // If addr is NULL, then the kernel chooses the (page-aligned) address at which to create the mapping
    if (gpio_ptr2 == MAP_FAILED) {
    	printf("Mmap call failure.\n");
    	return -1;
    }
    gpio_ptr3 = mmap(NULL, getpagesize(), PROT_READ|PROT_WRITE, MAP_SHARED, fd3, 0);
    // If addr is NULL, then the kernel chooses the (page-aligned) address at which to create the mapping
    if (gpio_ptr3 == MAP_FAILED) {
    	printf("Mmap call failure.\n");
    	return -1;
    }

    // Set bit0 on the GPIO to be output and bit 1 to be input
    *((volatile unsigned *)(gpio_ptr0 + GPIO_TRI)) = 0x00;
    *((volatile unsigned *)(gpio_ptr1 + GPIO_TRI)) = 0x00;
    *((volatile unsigned *)(gpio_ptr2 + GPIO_TRI)) = 0x00;
    *((volatile unsigned *)(gpio_ptr3 + GPIO_TRI)) = 0x01;

    // Assign start and lstm_input_vld input to 1
    *((volatile unsigned *)(gpio_ptr0 + GPIO_DATA)) = 0x01;
    *((volatile unsigned *)(gpio_ptr1 + GPIO_DATA)) = 0x01;

    printf("Opening a character device file of the Zynq DDR memory...\n");
    int ddr_memory = open("/dev/mem", O_RDWR | O_SYNC);
    if (ddr_memory == -1) {
        perror("/dev/mem could not be opened.\n");
        exit(1);
    } else {
    	printf("/dev/mem successfully opened.\n");
    }

    // Create memory map from source address register
    printf("Memory map the source address register block.\n");
    unsigned int* source_addr  = mmap(NULL, LENGTH, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_memory, SOURCE_ADDRESS); // SOURCE address
    if (source_addr == MAP_FAILED) {
        perror("mmap source_addr\n");
        exit(1);
    }
    else{
    	printf("Source address correctly mapped\n");
    }

	#ifdef INFERENCE_MODE

    while(1){

    	printf("Acquisition of data for inference mode . . .\n");

    	gettimeofday(&start, NULL);
		spi_data(source_addr);
		gettimeofday(&end, NULL);
		printf("Time taken for spi acquisition is : %ld micro seconds\n",
		    ((end.tv_sec * 1000000 + end.tv_usec) -
		    (start.tv_sec * 1000000 + start.tv_usec)));
		//create_logs(source_addr, "spi_data.dat");

		//load_data(sdmem, source_addr, save_data);

		// Normalization of saved data using train data as reference
		gettimeofday(&start, NULL);
		normalize_data(source_addr, save_data);
		gettimeofday(&end, NULL);
		printf("Time taken for normalization is : %ld micro seconds\n",
		    ((end.tv_sec * 1000000 + end.tv_usec) -
		    (start.tv_sec * 1000000 + start.tv_usec)));

		create_logs(source_addr, "normalized_data.dat");

		// Transformation of floating point data to fixed point
		gettimeofday(&start, NULL);
		to_fixed_point(save_data, source_addr);
		gettimeofday(&end, NULL);
		printf("Time taken for fixed point transformation is : %ld micro seconds\n",
		    ((end.tv_sec * 1000000 + end.tv_usec) -
		    (start.tv_sec * 1000000 + start.tv_usec)));

		printf("Transformation to fixed point completed!\n");

		i=0;

		// Hardware acceleration using AXI GPIOs
		gettimeofday(&start, NULL);
		for (i=0; i<MAXBUFF; i++){
			*((volatile unsigned *)(gpio_ptr2 + GPIO_DATA)) = source_addr[i];
			sleep(0.000000001);
			RECSTR = *((volatile signed *)(gpio_ptr3 + GPIO_DATA));

			if (RECSTR > LIM) // Condition for negative results
			{
				RECSTR = (RECSTR - LENGTH);
			}

			// Fixed point to floating point transformation
			RECSTR = RECSTR/256.0f;

			logs[i] = RECSTR;

			printf("For input %f, reconstructed value is %f\n", save_data[i], logs[i]);

		}
		gettimeofday(&end, NULL);
		printf("Time taken for reconstruction is : %ld micro seconds\n",
			((end.tv_sec * 1000000 + end.tv_usec) -
			(start.tv_sec * 1000000 + start.tv_usec)));

		create_logs(logs, "reconstructed_data.dat");

    }

	#else

   	i = 0;
   	char buffer[32];

   	// Acquisition of 20 datasets for training mode (327'680 datas)
   	for (i=0; i<20; i++){

		printf("Acquisition of data for training mode . . .\n");

		sprintf(buffer, "data_training_%i.dat", i);

		spi_data(source_addr);
		create_logs(source_addr, buffer);
   	}

	#endif

    // Unmap the AXI GPIOs devices and source address from user space
    munmap(gpio_ptr0, 4096);
    munmap(gpio_ptr1, 4096);
    munmap(gpio_ptr2, 4096);
    munmap(gpio_ptr3, 4096);
    munmap(source_addr, LENGTH);

    return 0;
}
