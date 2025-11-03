#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stdint.h>   
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "stb_image.h"
#include "stb_image_resize2.h"

#define BATCH 1
#define IN_C 3
#define IN_H 224  
#define IN_W 224
#define NUM_CLASSES 1000

#define IMAGENET_MEAN_R 0.485f
#define IMAGENET_MEAN_G 0.456f
#define IMAGENET_MEAN_B 0.406f
#define IMAGENET_STD_R  0.229f
#define IMAGENET_STD_G  0.224f
#define IMAGENET_STD_B  0.225f

typedef struct {
    float *allocated;
    float *aligned;
    int64_t offset;
    int64_t sizes[4];
    int64_t strides[4];
} MemRef4D;

extern void* alexnet(MemRef4D* input);

void memrefCopy(void) { }

static char* imagenet_classes[1000];
static int classes_loaded = 0;

static void load_imagenet_classes(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        char fallback_path[256];
        snprintf(fallback_path, sizeof(fallback_path), "../%s", filename);
        f = fopen(fallback_path, "r");
    }
    if (!f) {
        char data_path[256];
        snprintf(data_path, sizeof(data_path), "./data/%s", filename);
        f = fopen(data_path, "r");
    }
    if (!f) {
        fprintf(stderr, "Warning: Could not load class labels\n");
        return;
    }

    char line[256];
    int idx = 0;
    while (fgets(line, sizeof(line), f) && idx < 1000) {
        line[strcspn(line, "\n")] = 0;
        line[strcspn(line, "\r")] = 0;
        if (strlen(line) == 0) continue;
        char *comma = strchr(line, ',');
        if (comma) *comma = 0;
        imagenet_classes[idx] = strdup(line);
        idx++;
    }
    fclose(f);
    classes_loaded = idx;
}

static void cleanup_classes() {
    for (int i = 0; i < classes_loaded; i++) {
        if (imagenet_classes[i]) {
            free(imagenet_classes[i]);
            imagenet_classes[i] = NULL;
        }
    }
    classes_loaded = 0;
}

static const char* get_class_name(int class_id) {
    if (classes_loaded > 0 && class_id >= 0 && class_id < classes_loaded) {
        return imagenet_classes[class_id];
    }
    static char buf[32];
    snprintf(buf, sizeof(buf), "class_%d", class_id);
    return buf;
}

static int load_and_preprocess_image(const char *filepath, float *buffer) {
    int width, height, channels;
    unsigned char *img = stbi_load(filepath, &width, &height, &channels, 3);
    if (img == NULL) {
        fprintf(stderr, "Failed to load image '%s': %s\n", filepath, stbi_failure_reason());
        return -1;
    }

    unsigned char *resized = NULL;
    unsigned char *img_to_use = img;

    if (width != IN_W || height != IN_H) {
        resized = (unsigned char*)malloc(IN_W * IN_H * 3);
        if (resized == NULL) {
            stbi_image_free(img);
            return -1;
        }
        if (!stbir_resize_uint8_linear(img, width, height, 0, resized, IN_W, IN_H, 0, STBIR_RGB)) {
            free(resized);
            stbi_image_free(img);
            return -1;
        }
        img_to_use = resized;
    }

    const int total_pixels = IN_H * IN_W;

    {

        for (int i = 0; i < total_pixels; i++) {
            int h = i / IN_W;
            int w = i % IN_W;
            int pixel_idx = (h * IN_W + w) * 3;
            unsigned char r = img_to_use[pixel_idx + 0];
            float r_norm = r / 255.0f;
            float r_final = (r_norm - IMAGENET_MEAN_R) / IMAGENET_STD_R;
            buffer[i] = r_final;
        }

        for (int i = 0; i < total_pixels; i++) {
            int h = i / IN_W;
            int w = i % IN_W;
            int pixel_idx = (h * IN_W + w) * 3;
            unsigned char g = img_to_use[pixel_idx + 1];
            float g_norm = g / 255.0f;
            float g_final = (g_norm - IMAGENET_MEAN_G) / IMAGENET_STD_G;
            buffer[total_pixels + i] = g_final;
        }

        for (int i = 0; i < total_pixels; i++) {
            int h = i / IN_W;
            int w = i % IN_W;
            int pixel_idx = (h * IN_W + w) * 3;
            unsigned char b = img_to_use[pixel_idx + 2];
            float b_norm = b / 255.0f;
            float b_final = (b_norm - IMAGENET_MEAN_B) / IMAGENET_STD_B;
            buffer[2 * total_pixels + i] = b_final;
        }
    }

    if (resized) free(resized);
    stbi_image_free(img);
    return 0;
}

static void softmax(float *logits, float *probs, int num_classes) {

    float max_logit = logits[0];
    for (int i = 1; i < num_classes; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum = 0.0f;

    {
        float local_sum = 0.0f;

        for (int i = 0; i < num_classes; i++) {
            probs[i] = expf(logits[i] - max_logit);
            local_sum += probs[i];
        }

        sum += local_sum;
    }

    if (sum < 1e-10f) sum = 1e-10f;

    for (int i = 0; i < num_classes; i++) {
        probs[i] /= sum;
    }
}

static void get_topk(float *probs, int num_classes, int k, int *indices, float *values) {

    for (int i = 0; i < k; i++) {
        indices[i] = i;
        values[i] = probs[i];
    }

    for (int i = k/2 - 1; i >= 0; i--) {
        int parent = i;
        while (2 * parent + 1 < k) {
            int child = 2 * parent + 1;
            if (child + 1 < k && values[child + 1] < values[child]) {
                child++;
            }
            if (values[parent] <= values[child]) break;

            float tmp_val = values[parent];
            int tmp_idx = indices[parent];
            values[parent] = values[child];
            indices[parent] = indices[child];
            values[child] = tmp_val;
            indices[child] = tmp_idx;
            parent = child;
        }
    }

    for (int i = k; i < num_classes; i++) {
        if (probs[i] > values[0]) {
            values[0] = probs[i];
            indices[0] = i;

            int parent = 0;
            while (2 * parent + 1 < k) {
                int child = 2 * parent + 1;
                if (child + 1 < k && values[child + 1] < values[child]) {
                    child++;
                }
                if (values[parent] <= values[child]) break;

                float tmp_val = values[parent];
                int tmp_idx = indices[parent];
                values[parent] = values[child];
                indices[parent] = indices[child];
                values[child] = tmp_val;
                indices[child] = tmp_idx;
                parent = child;
            }
        }
    }

    for (int i = k - 1; i > 0; i--) {
        float tmp_val = values[0];
        int tmp_idx = indices[0];
        values[0] = values[i];
        indices[0] = indices[i];
        values[i] = tmp_val;
        indices[i] = tmp_idx;

        int parent = 0;
        while (2 * parent + 1 < i) {
            int child = 2 * parent + 1;
            if (child + 1 < i && values[child + 1] < values[child]) {
                child++;
            }
            if (values[parent] <= values[child]) break;

            tmp_val = values[parent];
            tmp_idx = indices[parent];
            values[parent] = values[child];
            indices[parent] = indices[child];
            values[child] = tmp_val;
            indices[child] = tmp_idx;
            parent = child;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image_path> [num_warmup_runs] [num_benchmark_runs]\n", argv[0]);
        return 1;
    }

    const char *image_path = argv[1];
    int num_warmup = (argc > 2) ? atoi(argv[2]) : 3;
    int num_benchmark = (argc > 3) ? atoi(argv[3]) : 10;


    load_imagenet_classes("../imagenet_classes.txt");

    size_t input_elems = (size_t)BATCH * IN_C * IN_H * IN_W;

    float *in_buf = NULL;
    if (posix_memalign((void**)&in_buf, 64, sizeof(float) * input_elems) != 0) {
        fprintf(stderr, "Failed to allocate input buffer\n");
        cleanup_classes();
        return 1;
    }

    MemRef4D input_desc;
    input_desc.allocated = in_buf;
    input_desc.aligned = in_buf;
    input_desc.offset = 0;
    input_desc.sizes[0] = BATCH;
    input_desc.sizes[1] = IN_C;
    input_desc.sizes[2] = IN_H;
    input_desc.sizes[3] = IN_W;
    input_desc.strides[0] = IN_C * IN_H * IN_W;
    input_desc.strides[1] = IN_H * IN_W;
    input_desc.strides[2] = IN_W;
    input_desc.strides[3] = 1;

    printf("Loading and preprocessing image...\n");
    double preprocess_start = omp_get_wtime();
    if (load_and_preprocess_image(image_path, in_buf) != 0) {
        free(in_buf);
        cleanup_classes();
        return 1;
    }
    double preprocess_time = (omp_get_wtime() - preprocess_start) * 1000.0;
    printf("Preprocessing time: %.3f ms\n", preprocess_time);

    printf("\nWarming up (%d runs)...\n", num_warmup);
    for (int i = 0; i < num_warmup; i++) {
        void* result = alexnet(&input_desc);
        if (result == NULL) {
            fprintf(stderr, "ERROR: alexnet returned NULL during warmup\n");
            free(in_buf);
            cleanup_classes();
            return 1;
        }
    }

    printf("Running benchmark (%d runs)...\n", num_benchmark);
    double total_time = 0.0;
    double min_time = INFINITY;
    double max_time = 0.0;
    void* final_result = NULL;

    for (int i = 0; i < num_benchmark; i++) {
        double start = omp_get_wtime();
        void* result = alexnet(&input_desc);
        double end = omp_get_wtime();

        double elapsed = (end - start) * 1000.0;
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
        if (elapsed > max_time) max_time = elapsed;

        if (i == num_benchmark - 1) {
            final_result = result;
        }
    }

    double avg_time = total_time / num_benchmark;

    printf("  Average: %.3f ms\n", avg_time);
    printf("  Min:     %.3f ms\n", min_time);
    printf("  Max:     %.3f ms\n", max_time);
    printf("  Throughput: %.2f FPS\n", 1000.0 / avg_time);

    if (final_result == NULL) {
        fprintf(stderr, "ERROR: alexnet returned NULL\n");
        free(in_buf);
        cleanup_classes();
        return 1;
    }

    float *out_buf = (float*)final_result;

    float sum_check = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        sum_check += fabsf(out_buf[i]);
    }

    if (sum_check < 1e-6f) {
        fprintf(stderr, "ERROR: All outputs are zero! Model did not run correctly.\n");
        fprintf(stderr, "First 10 outputs: ");
        for (int i = 0; i < 10; i++) {
            fprintf(stderr, "%.6f ", out_buf[i]);
        }
        fprintf(stderr, "\n");
        free(in_buf);
        cleanup_classes();
        return 1;
    }

    float *probs = (float*)calloc(NUM_CLASSES, sizeof(float));
    int *top_indices = (int*)calloc(5, sizeof(int));
    float *top_values = (float*)calloc(5, sizeof(float));

    if (!probs || !top_indices || !top_values) {
        fprintf(stderr, "Failed to allocate result buffers\n");
        if (probs) free(probs);
        if (top_indices) free(top_indices);
        if (top_values) free(top_values);
        free(in_buf);
        cleanup_classes();
        return 1;
    }

    softmax(out_buf, probs, NUM_CLASSES);
    get_topk(probs, NUM_CLASSES, 5, top_indices, top_values);

    printf("\n\nTop-5 Predictions\n\n");
    for (int i = 0; i < 5; i++) {
        if (top_indices[i] >= 0 && top_indices[i] < NUM_CLASSES) {
            printf("%d. Class %4d (%-30s): %.2f%%\n", 
                   i+1, 
                   top_indices[i],
                   get_class_name(top_indices[i]),
                   top_values[i] * 100.0f);
        }
    }

    free(probs);
    free(top_indices);
    free(top_values);
    free(in_buf);
    cleanup_classes();

    return 0;
}
