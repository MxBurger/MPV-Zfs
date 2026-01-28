# Thrust

## Codebeispiel Ambient Temp

### Sequentielle Lösung

```cpp
int steps = 3;
float k = 0.5;
float ambient_temp = 20;
std::vector<float> cups{42, 24, 50};

auto op = [=] (float t) {      // [=] captured Variablen per value aus dem umgebenden Scope
    float diff = ambient_temp - t;
    return t + k * diff;
};

for (int step : std::views::iota(0, steps)) // lazy Sequence [0, 1, 2]
{
    std::println("{} {}", step, cups);
    std::transform(cups.begin(),        // Beginn der Input-Elemente
                   cups.end(),          // Ende der Input-Elemente
                   cups.begin(),        // Beginn der Outputelemente
                   op                   // Transformationsfunktion
                );
}
```

###  Parallele Lösung mit `thrust::for_each_n`

```cpp
int steps = 3;
float k = 0.5;
float ambient_temp = 20;
thrust::universal_vector<float> cups{42, 24, 50};

auto op = [=] __host__ __device__ (float t) {
    float diff = ambient_temp - t;
    return t + k * diff;
};

for (int step : std::views::iota(0, steps))
{
    std::println("{} {}", step, cups);
    thrust::for_each_n(
        thrust::cuda::par,
        cups.begin(),
        cups.size(),
        op
    );
}
```

### Parallele Lösung mit `thrust::transform`

```cpp
int steps = 3;
float k = 0.5;
float ambient_temp = 20;
thrust::universal_vector<float> cups{42, 24, 50};  // implizit On-Demand kopierter Memory von Host ↔ Device

auto op = [=] __host__ __device__ (float t) {      // Funktion wird für CPU und GPU kompiliert
    float diff = ambient_temp - t;
    return t + k * diff;
};

for (int step : std::views::iota(0, steps))
{
    std::println("{} {}", step, cups);
    thrust::transform(thrust::cuda::par,        // Flag für parallele Verarbeitung
                      cups.begin(),             // Beginn der Input-Elemente
                      cups.end(),               // Ende der Input-Elemente
                      cups.begin(),             // Beginn der Outputelemente
                      op                        // Transformationsfunktion (muss auch auf GPU laufen)
                    );
}
```




## Iteratoren
### `thrust::make_counting_iterator()`

Iterator inkrementiert bei Abfrage

```cpp
int n = 5;
thrust::universal_vector<int> result(n);

// Quadratzahlen berechnen: 0², 1², 2², 3², 4²
auto square = [] __host__ __device__ (int i) {
    return i * i;
};

thrust::transform(
    thrust::cuda::par,
    thrust::make_counting_iterator(0),      // virtuelle Sequenz [0, 1, 2, ...]
    thrust::make_counting_iterator(n),      // Ende bei n (exklusiv)
    result.begin(),
    square
);

std::println("{}", result);  // [0, 1, 4, 9, 16]
```

```cpp
int n = 5;
thrust::universal_vector<int> result(n);

// i läuft von 0..n-1
thrust::for_each_n(
    thrust::cuda::par,
    thrust::make_counting_iterator<int>(0),
    n,
    [ptr = result.data()] __host__ __device__ (int i) {  // explizites capturen
        ptr[i] = i * i;
    }
);

std::println("{}", out); // [0, 1, 4, 9, 16]
```

### `thrust::make_constant_iterator()`

Liefert immer einen konstanten Wert

```cpp
thrust::universal_vector<float> values{10.f, 20.f, 30.f, 40.f};
thrust::universal_vector<float> result(values.size());

float offset = 42.0f;

auto add = [] __host__ __device__ (float a, float b) {
    return a + b;
};

thrust::transform(
    thrust::cuda::par,
    values.begin(),
    values.end(),
    thrust::make_constant_iterator(offset),  // liefert immer 42
    result.begin(),
    add
);

std::println("{}", result);  // [52, 62, 72, 82]
```


### `thrust::make_transform_iterator`

`transform_iterator` ist wie ein lazy map.

```cpp
thrust::universal_vector<float> cups{42.f, 24.f, 50.f};

float ambient = 20.f;
float k = 0.5f;

// Transformationsfunktion
auto cool = [=] __host__ __device__ (float t) {
    return t + k * (ambient - t);
};

// Iterator, der cups[i] virtuell transformiert
auto begin = thrust::make_transform_iterator(cups.begin(), cool);
auto end   = thrust::make_transform_iterator(cups.end(),   cool);

// Zielvektor
thrust::universal_vector<float> result(cups.size());

// Nutzung wie ein normaler Iterator
thrust::copy(
    thrust::cuda::par,
    begin,
    end,
    result.begin()
);

std::println("{}", result); // transformierte Werte
```

#### Beispiel mit `counting_iterator``

```cpp
int n = 5;
thrust::universal_vector<int> result(n);

auto f = [] __host__ __device__ (int i) {
    return i * i + 1;
};

thrust::copy(
    thrust::cuda::par,
    thrust::make_transform_iterator(
        thrust::make_counting_iterator(0), f
    ),
    thrust::make_transform_iterator(
        thrust::make_counting_iterator(n), f
    ),
    result.begin()
);

std::println("{}", result); // [1, 2, 5, 10, 17]
```

### `thrust::zip_iterator`

Sequenzen elementeweise bündeln

```cpp
thrust::universal_vector<int> a{1, 2, 3};
thrust::universal_vector<int> b{10, 20, 30};
thrust::universal_vector<int> result(a.size());

// Zip-Iterator über (a[i], b[i])
auto begin = thrust::make_zip_iterator(
    thrust::make_tuple(a.begin(), b.begin())
);
auto end = thrust::make_zip_iterator(
    thrust::make_tuple(a.end(), b.end())
);

thrust::transform(
    thrust::cuda::par,
    begin,
    end,
    result.begin(),
    [] __host__ __device__ (const thrust::tuple<int, int>& t) {
        return thrust::get<0>(t) + thrust::get<1>(t);
    }
);

std::println("{}", result); // [11, 22, 33]
```

## Standard Algorithmen

### `thrust::transform_reduce`

Ist wie ein `transform` + `reduce` ohne Zwischenspeicher.

```cpp
thrust::universal_vector<int> values{1, 2, 3, 4};

// Transformation: Quadrat
auto square = [] __host__ __device__ (int x) {
    return x * x;
};

// Reduktion: Summe
auto plus = thrust::plus<int>{};

int result = thrust::transform_reduce(
    thrust::cuda::par,
    values.begin(),
    values.end(),
    square,     // transform
    0,          // Initialwert der Reduktion
    plus        // reduce
);

std::println("{}", result); // 1 + 4 + 9 + 16 = 30
```

#### Mit `counting_iterator`

```cpp
int n = 5;

// Summe von i² für i = 0..4
int sum = thrust::transform_reduce(
    thrust::cuda::par,
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(n),
    [] __host__ __device__ (int i) { return i * i; },
    0,
    thrust::plus<int>{}
);

std::println("{}", sum); // 0 + 1 + 4 + 9 + 16 = 30
```


### `thrust::inclusive_scan`

Ist präfixweise Akkumulation

```cpp
thrust::universal_vector<int> values{1, 2, 3, 4};

thrust::inclusive_scan(
    thrust::cuda::par,      // Ausführungs-Policy
    values.begin(),         // Input-Beginn
    values.end(),           // Input-Ende
    values.begin()          // Output (in-place)
);

std::println("{}", values); // Ergebnis: [1, 3, 6, 10]

```

```
[1, 2, 3, 4]
 ↓
[1,
 1+2,
 1+2+3,
 1+2+3+4]
 ```

 #### mit eigener Operation

 ```cpp
 thrust::universal_vector<int> values{1, 2, 3, 4};

// Ergebnis: [1, 2, 6, 24]
thrust::inclusive_scan(
    thrust::cuda::par,
    values.begin(),
    values.end(),
    values.begin(),
    thrust::multiplies<int>{}
);

std::println("{}", values);
```

### `thrust::sort`

Sortiert die Sequenz

```cpp
thrust::universal_vector<int> values{4, 1, 3, 2};

thrust::sort(
    thrust::cuda::par,
    values.begin(),
    values.end(),
    thrust::greater<int>{}
);

std::println("{}", values); // [4, 3, 2, 1]
```

## Erweiterte Algorithmen

### `thrust::sort_by_key`

Sortiert sowohl keys als auch vals

```cpp
thrust::universal_vector<int> keys  {3, 1, 2};
thrust::universal_vector<char> vals {'c', 'a', 'b'};

// sortiert nach keys, vals werden mitgezogen
thrust::sort_by_key(
    thrust::cuda::par,
    keys.begin(),
    keys.end(),
    vals.begin()
);

std::println("{} {}", keys, vals); // keys: [1,2,3] vals: [a,b,c]
```

### `thrust::reduce_by_key`

Ist gruppieren nach Key und die Gruppen dann reduzieren.
Nur **benachbarte** gleiche Keys werden reduziert.

```cpp
thrust::universal_vector<int> keys   {1, 1, 2, 2, 2, 3};
thrust::universal_vector<int> values {10, 20,  1,  2,  3, 5};

thrust::universal_vector<int> out_keys(keys.size());
thrust::universal_vector<int> out_vals(values.size());

// Gruppiert nach key und summiert die values
auto result = thrust::reduce_by_key(
    thrust::cuda::par,
    keys.begin(),
    keys.end(),
    values.begin(),
    out_keys.begin(),
    out_vals.begin()
);

// Anzahl der erzeugten Gruppen
int n = result.first - out_keys.begin();

out_keys.resize(n);
out_vals.resize(n);

std::println("{} {}", out_keys, out_vals);
// keys:   [1, 2, 3]
// values: [30, 6, 5]
```

## mdspan

Ist eine non-owning View auf existierenden Speicher

```cpp
#include <mdspan>
#include <vector>
#include <print>

int main() {
    std::vector<int> data {1, 2, 3, 4, 5, 6};
    
    // Row-Major (layout_right) - C/C++ Standard
    std::mdspan<int, std::extents<size_t, 2, 3>, std::layout_right> row(data.data());
    
    // Column-Major (layout_left) - Fortran/MATLAB Style
    std::mdspan<int, std::extents<size_t, 2, 3>, std::layout_left> col(data.data());
    
    std::println("Speicher: [1, 2, 3, 4, 5, 6]\n");
    
    std::println("Row-Major (layout_right):");
    std::println("  Zeile 0: {} {} {}", row(0,0), row(0,1), row(0,2));
    std::println("  Zeile 1: {} {} {}", row(1,0), row(1,1), row(1,2));
    
    std::println("\nColumn-Major (layout_left):");
    std::println("  Zeile 0: {} {} {}", col(0,0), col(0,1), col(0,2));
    std::println("  Zeile 1: {} {} {}", col(1,0), col(1,1), col(1,2));
}
```

### Cuda Version

```cpp
float data[] = {1,2,3,4,5,6};

cuda::std::mdspan<
    float,
    cuda::std::extents<size_t, 2, 3>,
    cuda::std::layout_right
> row(data);

cuda::std::mdspan<
    float,
    cuda::std::extents<size_t, 2, 3>,
    cuda::std::layout_left
> col(data);

std::printf("Row-Major:\n");
std::printf("%f %f %f\n", row(0,0), row(0,1), row(0,2));
std::printf("%f %f %f\n", row(1,0), row(1,1), row(1,2));

std::printf("\nColumn-Major:\n");
std::printf("%f %f %f\n", col(0,0), col(0,1), col(0,2));
std::printf("%f %f %f\n", col(1,0), col(1,1), col(1,2));

```

```
Row-Major:
1.000000 2.000000 3.000000
4.000000 5.000000 6.000000

Column-Major:
1.000000 3.000000 5.000000
2.000000 4.000000 6.000000
```
