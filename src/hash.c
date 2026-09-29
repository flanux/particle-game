#include "pch.h"
#include "hash.h"

#include  "particle.h"

#include <immintrin.h>

Hash* ConstructHash(float s)
{
    Hash *hash = (Hash*)malloc(sizeof(Hash));
    memset(hash, 0, sizeof(*hash));

    hash->isCleared = true;
    hash->spacing   = s;
    hash->inv_spacing = 1.0f / s;

    hash->queryResults = NULL;
    arrsetcap(hash->queryResults, MAX_PARTICLE_COUNT);

    return hash;
}

void DestructHash(Hash *this)
{
    arrfree(this->queryResults);
    free(this);
}

void ClearHash(Hash *this)
{
    this->isCleared = true;

    const __m256i zero = _mm256_setzero_si256();
    for (size_t i = 0; i< CELL_COUNT; i += 8) {
        _mm256_storeu_si256((__m256i*)&this->cellCount[i], zero);
    }

    for (size_t i = 0; i< CELL_COUNT; i += 8) {
        _mm256_storeu_si256((__m256i*)&this->cellStart[i], zero);
    }


    arrsetlen(this->queryResults, 0);
}

void FillHash(Hash *this, const ParticlePool *particles)
{
    PASSERT(this->isCleared, LOG_WARNING, "Spatial Hash Map not cleared, before filling. ");
    if(!(this->isCleared)) { ClearHash(this); }

    float inv_s = this->inv_spacing;

    // count the total number of particles in each cell
    for(size_t i = 0; i < particles->activeCount; i++)
    {
        float x = particles->pPosX[i], y = particles->pPosY[i];
        // PASSERT((x > EPSILON && y > EPSILON), LOG_ERROR, "Particle position less than 0.");

        uint32_t cell = HashCoords_(
            (int)(particles->pPosX[i] * inv_s),
            (int)(particles->pPosY[i] * inv_s)
        );

        this->cellCount[cell] += 1;
    }

      // REVERTED TO SCALAR PREFIX SUM FOR DEBUGGING
    uint32_t partialSum = 0; 
    for(size_t i = 0; i < CELL_COUNT; i++)
    {
        partialSum += this->cellCount[i];
        this->cellStart[i] = partialSum;
    }

    /*
    __m256i running_sum = _mm256_setzero_si256();

    for (size_t i = 0; i < CELL_COUNT; i += 8) {

        __m256i counts = _mm256_loadu_si256((const __m256i*)&this->cellCount[i]);

        __m256i v1 = _mm256_slli_si256(counts, 4);
        __m256i sum1 = _mm256_add_epi32(counts, v1);

        __m256i v2 = _mm256_slli_si256(sum1, 8);
        __m256i sum2 = _mm256_add_epi32(sum1, v2);

        __m256i v3 = _mm256_slli_si256(sum2, 16);
        __m256i local_prefix = _mm256_add_epi32(sum2, v3);

        __m256i global_prefix = _mm256_add_epi32(local_prefix, running_sum);

        _mm256_storeu_si256((__m256i*)&this->cellStart[i], global_prefix);

        uint32_t last_val = _mm256_extract_epi32(global_prefix, 7);
        running_sum = _mm256_set1_epi32(last_val); 
    }
    */

    // Using the previously calculate partial sums to determine the index 
    // of each particle in the dense array of particles. When complete the cellStart 
    // array which previously contained the partial sums will contain the start index 
    // of cell in the dense array
    for(size_t i = 0; i < particles->activeCount; i++)
    {
        uint32_t cell = HashCoords_(
            (int)(particles->pPosX[i] * inv_s),
            (int)(particles->pPosY[i] * inv_s)
            );
        size_t index = --(this->cellStart[cell]);
        this->denseGrid[index] = i;
    }

    this->isCleared = false;
}

size_t QueryHashPoint(Hash *this, Vector2 position, float range)
{
    float xMin = position.x - range;
    float yMin = position.y - range;
    float xMax = position.x + range;
    float yMax = position.y + range;

    return QueryHashRange(this, xMin, xMax, yMin, yMax);
}

size_t QueryHashRange(Hash *this, float xMin, float xMax, float yMin, float yMax)
{
    PASSERT((xMin <= xMax), LOG_WARNING, "Spatial hash query invalid range. x-max is less than x-min.");
    PASSERT((yMin <= yMax), LOG_WARNING, "Spatial hash query invalid range. y-max is less than y-min.");

    arrsetlen(this->queryResults, 0);

    int x0 = CalculateCellCoord_(xMin, this->spacing);
    int y0 = CalculateCellCoord_(yMin, this->spacing);

    int x1 = CalculateCellCoord_(xMax, this->spacing);
    int y1 = CalculateCellCoord_(yMax, this->spacing);

    for(int xi = x0; xi <= x1; xi++)
    {
        for(int yi = y0; yi <= y1; yi++)
        {
            size_t h = HashCoords_(xi, yi);

            size_t start = this->cellStart[h];
            size_t end = start + this->cellCount[h];
            PASSERT((start <= end), LOG_ERROR, "end index is less than start index");
        
            for(size_t i = start; i < end; i++)
            {
                arrput(this->queryResults, this->denseGrid[i]);
            }
        }
    }
    return arrlenu(this->queryResults);
}
