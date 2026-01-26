
// Device-Kernel zur Umwandlung eines RGB-Bildes in Graustufen
__global__
void grayscale_kernel(unsigned char* red, 
                      unsigned char* green, 
                      unsigned char* blue,
                      unsigned char* gray,
                      unsigned int width, 
                      unsigned int height) {
    
    // Schritt 1: Globale 2D-Position berechnen
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Schritt 2: Boundary Check (für beide Dimensionen)
    if (row < height && col < width) {
        
        // Schritt 3: 2D-Koordinaten → 1D-Index (Linearisierung)
        unsigned int i = row * width + col;
        
        // Schritt 4: Graustufen-Berechnung
        gray[i] = red[i] * 3/10 + green[i] * 6/10 + blue[i] * 1/10;
    }
}


// Host-Code zur Verwendung des Grayscale-Kernels
void convert_to_grayscale(unsigned char* h_red,
                          unsigned char* h_green,
                          unsigned char* h_blue,
                          unsigned char* h_gray,
                          unsigned int width,
                          unsigned int height) {
    
    size_t size = width * height * sizeof(unsigned char);
    
    // 1. GPU-Speicher allokieren
    unsigned char *d_red, *d_green, *d_blue, *d_gray;
    cudaMalloc(&d_red, size);
    cudaMalloc(&d_green, size);
    cudaMalloc(&d_blue, size);
    cudaMalloc(&d_gray, size);
    
    // 2. Daten zur GPU kopieren
    cudaMemcpy(d_red, h_red, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_green, h_green, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_blue, h_blue, size, cudaMemcpyHostToDevice);
    
    // 3. Kernel konfigurieren und starten
    dim3 threadsPerBlock(32, 32);  // 1024 Threads pro Block
    dim3 numBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    
    grayscale_kernel<<<numBlocks, threadsPerBlock>>>(
        d_red, d_green, d_blue, d_gray, width, height
    );
    
    // 4. Ergebnis zurückkopieren
    cudaMemcpy(h_gray, d_gray, size, cudaMemcpyDeviceToHost);
    
    // 5. Aufräumen
    cudaFree(d_red);
    cudaFree(d_green);
    cudaFree(d_blue);
    cudaFree(d_gray);
}
