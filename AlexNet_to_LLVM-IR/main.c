#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize2.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

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

extern void alexnet(float**, float**);

void memrefCopy(void) { }

static char* imagenet_classes[1000];
static int classes_loaded = 0;




/*static void load_imagenet_classes(const char *filename) {
    FILE *f = NULL;

    f = fopen(filename, "r");

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

        fprintf(stderr, "Cautionn!!!: Sorry dude the path could be wrong.\n");
        fprintf(stderr, "Try once with correct path instead\n");
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
    printf("Yayy the number of classes loaded  are:  %d\n\n", classes_loaded);
}
*/

static void load_imagenet_classes(const char *filename) {
    FILE *f = NULL;

    f = fopen(filename, "r");

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

        fprintf(stderr, "Cautionn!!!: Sorry dude the path could be wrong.\n");
        fprintf(stderr, "Try once with correct path instead\n");
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
    printf("Loaded %d ImageNet class labels\n\n", classes_loaded);
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

    printf("Loading image: %s\n", filepath);

    unsigned char *img = stbi_load(filepath, &width, &height, &channels, 3);
    if (img == NULL) {
        fprintf(stderr, "Cautionn!!!: Sorry the path could be wrong. '%s'\n", filepath);
        fprintf(stderr, "and the reason is : %s\n", stbi_failure_reason());
        return -1;
    }

    printf("Original image: %dx%d pixels, %d channels\n", width, height, channels);

    unsigned char *resized = NULL;
    unsigned char *img_to_use = img;

    if (width != IN_W || height != IN_H) {
        printf("Resizing to %dx%d...\n", IN_W, IN_H);
        resized = (unsigned char*)malloc(IN_W * IN_H * 3);
        if (resized == NULL) {
            fprintf(stderr, "The system is unable to allocate resize buffer\n");
            stbi_image_free(img);
            return -1;
        }

        if (!stbir_resize_uint8_linear(img, width, height, 0,
                                       resized, IN_W, IN_H, 0,
                                       STBIR_RGB)) {
            fprintf(stderr, "Failed to resize image\n");
            free(resized);
            stbi_image_free(img);
            return -1;
        }

        img_to_use = resized;
    }

    printf("Take a breakk!!!! Have a KITKAT.... Normalization in progress......\n");

    for (int h = 0; h < IN_H; h++) {
        for (int w = 0; w < IN_W; w++) {
            int pixel_idx = (h * IN_W + w) * 3;

            unsigned char r = img_to_use[pixel_idx + 0];
            unsigned char g = img_to_use[pixel_idx + 1];
            unsigned char b = img_to_use[pixel_idx + 2];

            float r_norm = r / 255.0f;
            float g_norm = g / 255.0f;
            float b_norm = b / 255.0f;

            float r_final = (r_norm - IMAGENET_MEAN_R) / IMAGENET_STD_R;
            float g_final = (g_norm - IMAGENET_MEAN_G) / IMAGENET_STD_G;
            float b_final = (b_norm - IMAGENET_MEAN_B) / IMAGENET_STD_B;

            int r_offset = 0;
            int g_offset = IN_H * IN_W;
            int b_offset = 2 * IN_H * IN_W;

            buffer[r_offset + h * IN_W + w] = r_final;
            buffer[g_offset + h * IN_W + w] = g_final;
            buffer[b_offset + h * IN_W + w] = b_final;
        }
    }

    if (resized) {
        free(resized);
    }
    stbi_image_free(img);  

    printf("\n\n Funtime is over. Normalization completed :-) \n\n");
    return 0;
}

static void softmax(float *logits, float *probs, int num_classes) {
    float max_logit = logits[0];
    for (int i = 1; i < num_classes; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }

    if (sum < 1e-10f) sum = 1e-10f;

    for (int i = 0; i < num_classes; i++) {
        probs[i] /= sum;
    }
}

static void get_topk(float *probs, int num_classes, int k, int *indices, float *values) {

    for (int i = 0; i < k; i++) {
        indices[i] = -1;
        values[i] = -INFINITY;
    }

    for (int i = 0; i < k; i++) {
        int max_idx = 0;
        float max_val = -INFINITY;

        for (int j = 0; j < num_classes; j++) {
            int already_selected = 0;
            for (int m = 0; m < i; m++) {
                if (indices[m] == j) {
                    already_selected = 1;
                    break;
                }
            }

            if (!already_selected && probs[j] > max_val) {
                max_val = probs[j];
                max_idx = j;
            }
        }

        indices[i] = max_idx;
        values[i] = max_val;
    }
}

static void debug_output(float *output, int num_classes) {
    printf("\nDebug: First 10 raw logits:\n");
    for (int i = 0; i < 10 && i < num_classes; i++) {
        printf("  [%d]: %.6f\n", i, output[i]);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image_path>\n", argv[0]);
        fprintf(stderr, "Example: %s dog.jpg\n", argv[0]);
        return 1;
    }

    const char *image_path = argv[1];

    printf("\nI am AlexNet and i was highly influential that popularized the use of neural networks\n");
    printf("Input: %dx%dx%d\n", IN_H, IN_W, IN_C);
    printf("Classes: %d\n\n", NUM_CLASSES);

    size_t input_elems = (size_t)BATCH * IN_C * IN_H * IN_W;
    size_t output_elems = (size_t)BATCH * NUM_CLASSES;

    load_imagenet_classes("../imagenet_classes.txt");

    float *in_buf = NULL;
    float *out_buf = NULL;

    if (posix_memalign((void**)&in_buf, 64, sizeof(float) * input_elems) != 0) {
        fprintf(stderr, "UFF :-( failed to allocate input buffer\n");
        cleanup_classes();
        return 1;
    }

    if (posix_memalign((void**)&out_buf, 64, sizeof(float) * output_elems) != 0) {
        fprintf(stderr, "Umm :-( failed to allocate output buffer\n");
        free(in_buf);
        cleanup_classes();
        return 1;
    }

    memset(in_buf, 0, sizeof(float) * input_elems);
    memset(out_buf, 0, sizeof(float) * output_elems);

    if (load_and_preprocess_image(image_path, in_buf) != 0) {
        free(in_buf);
        free(out_buf);
        cleanup_classes();
        return 1;
    }

    printf("You can chill again.... Inferencing is progress!!!!!!\n");
double start_time  = clock();
    alexnet(&out_buf, &in_buf);
double end_time  = clock();
double time_taken = (end_time - start_time) / CLOCKS_PER_SEC;
printf("Time taken for inference: %f seconds\n", time_taken);

    printf("\nBreak over. Run another test........Inference completed!!!!!\n\n");

    float sum_check = 0.0f;
    int non_zero_count = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        sum_check += fabsf(out_buf[i]);
        if (fabsf(out_buf[i]) > 1e-6f) non_zero_count++;
    }

    if (sum_check < 1e-6f) {
        fprintf(stderr, "WARNING: All outputs are zero! Model may not have run correctly.\n");
        debug_output(out_buf, NUM_CLASSES);
        free(in_buf);
        free(out_buf);
        cleanup_classes();
        return 1;
    }

   
    printf("Output Statistics:\n");
    float min_val = out_buf[0], max_val = out_buf[0], sum_val = 0.0f;

    for (int i = 0; i < NUM_CLASSES; i++) {
        if (out_buf[i] < min_val) min_val = out_buf[i];
        if (out_buf[i] > max_val) max_val = out_buf[i];
        sum_val += out_buf[i];
    }

    // printf("  Logit range: [%.6f, %.6f]\n", min_val, max_val);
    // printf("  Mean logit: %.6f\n", sum_val / NUM_CLASSES);
    // printf("  Sum of logits: %.6f\n", sum_val);
    // printf("  Non-zero outputs: %d/%d\n\n", non_zero_count, NUM_CLASSES);

    float *probs = (float*)calloc(NUM_CLASSES, sizeof(float));
    int *top_indices = (int*)calloc(10, sizeof(int));
    float *top_values = (float*)calloc(10, sizeof(float));

    if (!probs || !top_indices || !top_values) {
        fprintf(stderr, "Failed to allocate result buffers\n");
        if (probs) free(probs);
        if (top_indices) free(top_indices);
        if (top_values) free(top_values);
        free(in_buf);
        free(out_buf);
        cleanup_classes();
        return 1;
    }

    softmax(out_buf, probs, NUM_CLASSES);
    get_topk(probs, NUM_CLASSES, 10, top_indices, top_values);

   
    printf("\nTop-10 Predictions:\n");
   
    for (int i = 0; i < 10; i++) {
        if (top_indices[i] >= 0 && top_indices[i] < NUM_CLASSES) {
            printf("%2d. Class %4d (%-30s): %.4f (%.2f%%)\n", 
                   i+1, 
                   top_indices[i],
                   get_class_name(top_indices[i]),
                   top_values[i], 
                   top_values[i] * 100.0f);
        }
    }

    float top1_confidence = top_values[0] * 100.0f;
    float top5_confidence = 0.0f;
    for (int i = 0; i < 5; i++) {
        top5_confidence += top_values[i];
    }
    top5_confidence *= 100.0f;

    printf("\n\nNot so confidence Metrics:\n\n");
    printf("  Top-1 confidence: %.2f%%\n", top1_confidence);
    printf("  Top-5 confidence: %.2f%%\n\n", top5_confidence);

    if (top1_confidence > 50.0f) {
        printf("  Too much confidence prediction!\n");
    } else if (top1_confidence > 20.0f) {
        printf("  moderate  confidence\n");
    } else {
        printf(" No confidence - image is not matching the training data\n");
    }

    printf("                Everythin is over!!!!                     \n");

    
    return 0;
}
