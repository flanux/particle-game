#include "pch.h"
#include "particle.h"
#include <immintrin.h>

#include "hash.h"

ParticleProps defaultParticleProps = {
    0.5f,                   // varaince
    1000.0f,                 // lifetime
    { 0.0f, 250.0f },      // velocity
    1.0f,                  // mass
};

// Set up vertex data - unit square (±0.5), scaled in shader
static const float quadVertices[] = {
    // positions    // texCoords
    -0.5f,  0.5f, 0.0f, 1.0f,
    -0.5f, -0.5f, 0.0f, 0.0f,
     0.5f, -0.5f, 1.0f, 0.0f,

    -0.5f,  0.5f, 0.0f, 1.0f,
     0.5f, -0.5f, 1.0f, 0.0f,
     0.5f,  0.5f, 1.0f, 1.0f,
};
static uint32_t quadVAO, quadVBO, instancePositionVBO, shaderId; //particle render state
static Vector2 instancePositions[MAX_PARTICLE_COUNT];

static ParticlePool* ConstructParticlePool_() 
{
    ParticlePool *particles = (ParticlePool*)malloc(sizeof(ParticlePool));
    PASSERT(particles, LOG_FATAL, "Failed to allocate particle particles");
    if(!particles) { return NULL; }

    memset(particles, 0, sizeof(*particles));

    particles->activeCount = 0;

    return particles;
}

static void DestructParticlePool_(ParticlePool *particles) 
{
    free(particles);
}

static void SwapParticles_(ParticlePool *particles, size_t i, size_t j)
{
    particles->pLifetimes[i] = particles->pLifetimes[j];
    particles->pLifespans[i] = particles->pLifespans[j];

    particles->pPrevPosX[i] = particles->pPrevPosX[j];
    particles->pPrevPosY[i] = particles->pPrevPosY[j];

    particles->pPosX[i] = particles->pPosX[j];
    particles->pPosY[i] = particles->pPosY[j];

    particles->pVelX[i] = particles->pVelX[j];
    particles->pVelY[i] = particles->pVelY[j];

    particles->pMasses[i] = particles->pMasses[j];
}

static void KillParticle_(ParticlePool *particles, size_t index) 
{
    particles->activeCount--;
    SwapParticles_(particles, index, particles->activeCount);
}

void ProjectSelfCollision(const Constraint *this, ParticlePool *particles, float deltaTime)
{
    PASSERTRETURN(this->participantCount == 2, LOG_WARNING, 
        "Incorrect number of participants in self collision constraint. Constraint participants must equal 2.");

    const size_t i = this->participants[0], j = this->participants[1];
    Vector2 pi = {
        particles->pPosX[i],
        particles->pPosY[i]
    };

    Vector2 pj = {
        particles->pPosX[j],
        particles->pPosY[j]
    };

    Vector2 seperation  = Vector2Subtract(pj, pi);
    float distance      = Vector2Length(seperation);
    Vector2 gradientC   = Vector2Normalize(seperation);
    
    float restLength      = 2.0f * PARTICLE_RADIUS;
    float constraintEval  = (distance - restLength);
    float iInvMass        = 1.0f / particles->pMasses[i], jInvMass = 1.0f / particles->pMasses[j];
    
    float lambda = constraintEval / (iInvMass + jInvMass);
    
    // Clamp maximum displacement to prevent instability
    // Maximum displacement per iteration should not exceed particle radius / substeps
    float maxDisplacement = PARTICLE_RADIUS / (float)PHYSICS_SUBSTEPS;
    float maxLambda = maxDisplacement / fmaxf(iInvMass, jInvMass);
    lambda = Clamp(lambda, -maxLambda, maxLambda);

    Vector2 deltaPi = Vector2Scale( gradientC, (lambda * iInvMass));
    Vector2 deltaPj = Vector2Scale( gradientC, (-1.0f * lambda * jInvMass));

    particles->pPosX[i] = pi.x + deltaPi.x;
    particles->pPosY[i] = pi.y + deltaPi.y;

    particles->pPosX[j] = pj.x + deltaPj.x;
    particles->pPosY[j] = pj.y + deltaPj.y;
}

void ProjectDistance(const Constraint *this, ParticlePool *particles, float deltaTime)
{
    PASSERT(false, LOG_WARNING, "ProjectDistance function not implemented");
}

static Vector2 CalculateForces_(
    const ForcePool *forces,
    Vector2 pi,
    Vector2 vi,
    float mi
)
{
    Vector2 externalForces = (Vector2){ 0 };

    // Gravity.
    externalForces.y +=
        mi * GRAVITIONAL_CONST * (float)forces->gravityCount;

    // Viscous forces.
    float totalViscosity = 0.0f;

    for (size_t i = 0; i < forces->viscosityCount; i++)
    {
        totalViscosity += forces->viscosity[i];
    }

    const float viscosityScale =
        -6.0f * PI * totalViscosity * PARTICLE_RADIUS;

    externalForces.x += vi.x * viscosityScale;
    externalForces.y += vi.y * viscosityScale;

    // Attractors.
    for (size_t i = 0; i < forces->attractCount; i++)
    {
        const float directionX =
            forces->attractPosX[i] - pi.x;

        const float directionY =
            forces->attractPosY[i] - pi.y;

        const float distanceSqr =
            directionX * directionX +
            directionY * directionY;

        if (distanceSqr < 1.0f)
        {
            continue;
        }

        const float distance = sqrtf(distanceSqr);

        const float normalizedX =
            directionX / distance;

        const float normalizedY =
            directionY / distance;

        const float strength =
            (mi * forces->attractMass[i]) /
            (distanceSqr + 10.0f);

        externalForces.x += normalizedX * strength;
        externalForces.y += normalizedY * strength;
    }

    // Repulsors.
    for (size_t i = 0; i < forces->repulseCount; i++)
    {
        const float directionX =
            forces->repulsePosX[i] - pi.x;

        const float directionY =
            forces->repulsePosY[i] - pi.y;

        const float distanceSqr =
            directionX * directionX +
            directionY * directionY;

        if (distanceSqr < 1.0f)
        {
            continue;
        }

        const float distance = sqrtf(distanceSqr);

        const float normalizedX =
            directionX / distance;

        const float normalizedY =
            directionY / distance;

        const float strength =
            -(mi * forces->repulseMass[i]) /
            (distanceSqr + 10.0f);

        externalForces.x += normalizedX * strength;
        externalForces.y += normalizedY * strength;
    }

    PASSERT(
        isfinite(externalForces.x) &&
        isfinite(externalForces.y),
        LOG_ERROR,
        "externalForces invalid."
    );

    return externalForces;
}

static size_t GenerateCollisionConstraints_(ParticleSystem *system)
{
    size_t collisionCount = 0;

    // Check for particle self collision
    const float range = 2.0f * PARTICLE_RADIUS;
    const float collisionGracePeriod = 0.05f; // Skip collision for newly spawned particles
    
    for (size_t i = 0; i < system->particles_->activeCount; i++)
    {
        // Skip collision detection for particles in grace period
        if (system->particles_->pLifespans[i] < collisionGracePeriod) { continue; }

        Vector2 pi = {
            system->particles_->pPosX[i],
            system->particles_->pPosY[i]
        };
        
        QueryHashPoint(system->spatialHash,
                       pi,
                       2.0f * PARTICLE_RADIUS);
        for (size_t j = 0; j < arrlenu(system->spatialHash->queryResults); j++)
        {
            size_t particleIndex = system->spatialHash->queryResults[j];

            if (i >= particleIndex) { continue; }

            if (system->particles_->pLifespans[particleIndex] < collisionGracePeriod)
            {
                continue;
            }

            Vector2 particlePosition = {
                system->particles_->pPosX[particleIndex],
                system->particles_->pPosY[particleIndex]
            };

            float dist = Vector2Distance(pi, particlePosition);

            float minDistance = 1e-3f;
            if (minDistance < dist && dist < range)
            {
                AddSelfCollisionConstraint(system, i, particleIndex);
                collisionCount++;
            }
        }
    }
    return collisionCount;
}

static void HandleBoundaryCollisions_(ParticleSystem *system)
{
    ParticlePool *particles = system->particles_;
    const float restitution = 0.7f;
    const float friction = 0.01f;

    for (size_t i = 0; i < particles->activeCount; i++)
    {
        float *posX = &particles->pPosX[i];
        float *posY = &particles->pPosY[i];

        if (*posX < system->boundaryBox.left + PARTICLE_RADIUS)
        {
            *posX = system->boundaryBox.left + PARTICLE_RADIUS;
            particles->pVelX[i] *= -restitution;
            particles->pVelY[i] *= (1.0f - friction);
        }
        else if (system->boundaryBox.right - PARTICLE_RADIUS < *posX)
        {
            *posX = system->boundaryBox.right - PARTICLE_RADIUS;
            particles->pVelX[i] *= -restitution;
            particles->pVelY[i] *= (1.0f - friction);
        }

        if (*posY < system->boundaryBox.bottom + PARTICLE_RADIUS)
        {
            *posY = system->boundaryBox.bottom + PARTICLE_RADIUS;
            particles->pVelX[i] *= (1.0f - friction);
            particles->pVelY[i] *= -restitution;
        }
        else if (system->boundaryBox.top - PARTICLE_RADIUS < *posY)
        {
            *posY = system->boundaryBox.top - PARTICLE_RADIUS;
            particles->pVelX[i] *= (1.0f - friction);
            particles->pVelY[i] *= -restitution;
        }
    }
}

static void UpdateParticlesLife_(ParticleSystem *system, float deltaTime)
{
    // Update lifespan of particles and deactivate/kill any particles whose
    // lifespan has exceeded its lifetime.
    size_t deadCount = 0;
    for (size_t i = 0; i < system->particles_->activeCount; i++) 
    {
        system->particles_->pLifespans[i] += deltaTime;
        if (system->particles_->pLifespans[i] > system->particles_->pLifetimes[i])
        { 
            deadCount++;
            SwapParticles_(system->particles_, i, (system->particles_->activeCount - deadCount));
        }
    }
    system->particles_->activeCount -= deadCount;
}

static void UpdateParticleAttributes_(ParticleSystem *system)
{
    return;
}

static void IntegrateVerlet_(ParticleSystem *system, float deltaTime)
{

    ParticlePool *particles = system->particles_;

    for (size_t i = 0; i < system->particles_->activeCount; i++)
    {
        Vector2 position = {
            particles->pPosX[i],
            particles->pPosY[i]
        };

        Vector2 velocity = {
            particles->pVelX[i],
            particles->pVelY[i]
        };

        Vector2 forces = CalculateForces_(&system->forces_,
                position,
                velocity,
                particles->pMasses[i]
        );

        particles->pPrevPosX[i] = particles->pPosX[i];
        particles->pPrevPosY[i] = particles->pPosY[i];

        particles->pPosX[i] +=
         particles->pVelX[i] * deltaTime + 
         forces.x * (deltaTime * deltaTime / particles->pMasses[i]);

        particles->pPosY[i] +=
         particles->pVelY[i] * deltaTime + 
         forces.y * (deltaTime * deltaTime / particles->pMasses[i]);
    }
}

static void IntegrateEuler_(
    ParticleSystem *system,
    float deltaTime
)
{
    ParticlePool *particles = system->particles_;
    const ForcePool *forces = &system->forces_;

    const size_t particleCount =
        particles->activeCount;

    float totalViscosity = 0.0f;

    for (size_t j = 0; j < forces->viscosityCount; j++)
    {
        totalViscosity += forces->viscosity[j];
    }

    const float gravityScale =
        GRAVITIONAL_CONST * (float)forces->gravityCount;

    const __m256 dt =
        _mm256_set1_ps(deltaTime);

    const __m256 gravity =
        _mm256_set1_ps(gravityScale);

    const __m256 viscosity =
        _mm256_set1_ps(
            -6.0f *
            PI *
            totalViscosity *
            PARTICLE_RADIUS
        );

    const __m256 one =
        _mm256_set1_ps(1.0f);

    const __m256 softening =
        _mm256_set1_ps(10.0f);

    for (size_t i = 0;
         i + 8 <= particleCount;
         i += 8)
    {
        /*
         * Load particle state.
         */

        const __m256 posX =
            _mm256_loadu_ps(
                &particles->pPosX[i]
            );

        const __m256 posY =
            _mm256_loadu_ps(
                &particles->pPosY[i]
            );

        __m256 velX =
            _mm256_loadu_ps(
                &particles->pVelX[i]
            );

        __m256 velY =
            _mm256_loadu_ps(
                &particles->pVelY[i]
            );

        const __m256 mass =
            _mm256_loadu_ps(
                &particles->pMasses[i]
            );

        /*
         * Accumulated external force.
         */

        __m256 forceX =
            _mm256_setzero_ps();

        __m256 forceY =
            _mm256_setzero_ps();

        /*
         * Gravity.
         */

        if (forces->gravityCount > 0)
        {
            forceY =
                _mm256_mul_ps(
                    mass,
                    gravity
                );
        }

        /*
         * Viscosity.
         */

        if (totalViscosity != 0.0f)
        {
            forceX =
                _mm256_add_ps(
                    forceX,
                    _mm256_mul_ps(
                        velX,
                        viscosity
                    )
                );

            forceY =
                _mm256_add_ps(
                    forceY,
                    _mm256_mul_ps(
                        velY,
                        viscosity
                    )
                );
        }

        /*
         * Attractors.
         */

        for (size_t j = 0;
             j < forces->attractCount;
             j++)
        {
            const __m256 forcePosX =
                _mm256_set1_ps(
                    forces->attractPosX[j]
                );

            const __m256 forcePosY =
                _mm256_set1_ps(
                    forces->attractPosY[j]
                );

            const __m256 forceMass =
                _mm256_set1_ps(
                    forces->attractMass[j]
                );

            const __m256 directionX =
                _mm256_sub_ps(
                    forcePosX,
                    posX
                );

            const __m256 directionY =
                _mm256_sub_ps(
                    forcePosY,
                    posY
                );

            const __m256 distanceSqr =
                _mm256_add_ps(
                    _mm256_mul_ps(
                        directionX,
                        directionX
                    ),
                    _mm256_mul_ps(
                        directionY,
                        directionY
                    )
                );

            const __m256 validMask =
                _mm256_cmp_ps(
                    distanceSqr,
                    one,
                    _CMP_GE_OQ
                );

            const __m256 safeDistanceSqr =
                _mm256_max_ps(
                    distanceSqr,
                    one
                );

            const __m256 distance =
                _mm256_sqrt_ps(
                    safeDistanceSqr
                );

            const __m256 normalizedX =
                _mm256_div_ps(
                    directionX,
                    distance
                );

            const __m256 normalizedY =
                _mm256_div_ps(
                    directionY,
                    distance
                );

            const __m256 strength =
                _mm256_div_ps(
                    _mm256_mul_ps(
                        mass,
                        forceMass
                    ),
                    _mm256_add_ps(
                        distanceSqr,
                        softening
                    )
                );

            const __m256 contributionX =
                _mm256_mul_ps(
                    normalizedX,
                    strength
                );

            const __m256 contributionY =
                _mm256_mul_ps(
                    normalizedY,
                    strength
                );

            forceX =
                _mm256_add_ps(
                    forceX,
                    _mm256_and_ps(
                        contributionX,
                        validMask
                    )
                );

            forceY =
                _mm256_add_ps(
                    forceY,
                    _mm256_and_ps(
                        contributionY,
                        validMask
                    )
                );
        }

        /*
         * Repulsors.
         */

        for (size_t j = 0;
             j < forces->repulseCount;
             j++)
        {
            const __m256 forcePosX =
                _mm256_set1_ps(
                    forces->repulsePosX[j]
                );

            const __m256 forcePosY =
                _mm256_set1_ps(
                    forces->repulsePosY[j]
                );

            const __m256 forceMass =
                _mm256_set1_ps(
                    forces->repulseMass[j]
                );

            const __m256 directionX =
                _mm256_sub_ps(
                    forcePosX,
                    posX
                );

            const __m256 directionY =
                _mm256_sub_ps(
                    forcePosY,
                    posY
                );

            const __m256 distanceSqr =
                _mm256_add_ps(
                    _mm256_mul_ps(
                        directionX,
                        directionX
                    ),
                    _mm256_mul_ps(
                        directionY,
                        directionY
                    )
                );

            const __m256 validMask =
                _mm256_cmp_ps(
                    distanceSqr,
                    one,
                    _CMP_GE_OQ
                );

            const __m256 safeDistanceSqr =
                _mm256_max_ps(
                    distanceSqr,
                    one
                );

            const __m256 distance =
                _mm256_sqrt_ps(
                    safeDistanceSqr
                );

            const __m256 normalizedX =
                _mm256_div_ps(
                    directionX,
                    distance
                );

            const __m256 normalizedY =
                _mm256_div_ps(
                    directionY,
                    distance
                );

            const __m256 strength =
                _mm256_div_ps(
                    _mm256_mul_ps(
                        mass,
                        forceMass
                    ),
                    _mm256_add_ps(
                        distanceSqr,
                        softening
                    )
                );

            const __m256 contributionX =
                _mm256_mul_ps(
                    normalizedX,
                    strength
                );

            const __m256 contributionY =
                _mm256_mul_ps(
                    normalizedY,
                    strength
                );

            forceX =
                _mm256_sub_ps(
                    forceX,
                    _mm256_and_ps(
                        contributionX,
                        validMask
                    )
                );

            forceY =
                _mm256_sub_ps(
                    forceY,
                    _mm256_and_ps(
                        contributionY,
                        validMask
                    )
                );
        }

        /*
         * F = ma
         *
         * dv = F / m * dt
         */

        const __m256 inverseMass =
            _mm256_div_ps(
                dt,
                mass
            );

        velX =
            _mm256_add_ps(
                velX,
                _mm256_mul_ps(
                    forceX,
                    inverseMass
                )
            );

        velY =
            _mm256_add_ps(
                velY,
                _mm256_mul_ps(
                    forceY,
                    inverseMass
                )
            );

        /*
         * Integrate position.
         */

        const __m256 deltaX =
            _mm256_mul_ps(
                velX,
                dt
            );

        const __m256 deltaY =
            _mm256_mul_ps(
                velY,
                dt
            );

        const __m256 newPosX =
            _mm256_add_ps(
                posX,
                deltaX
            );

        const __m256 newPosY =
            _mm256_add_ps(
                posY,
                deltaY
            );

        /*
         * Store velocity.
         */

        _mm256_storeu_ps(
            &particles->pVelX[i],
            velX
        );

        _mm256_storeu_ps(
            &particles->pVelY[i],
            velY
        );

        /*
         *  Store previous position
         */
        _mm256_storeu_ps(&particles->pPrevPosX[i], posX);   
        _mm256_storeu_ps(&particles->pPrevPosY[i], posY);   


        /*
         * Store position.
         */

        _mm256_storeu_ps(
            &particles->pPosX[i],
            newPosX
        );

        _mm256_storeu_ps(
            &particles->pPosY[i],
            newPosY
        );
    }

    /*
     * Scalar remainder.
     */

    for (size_t i = particleCount & ~((size_t)7);
         i < particleCount;
         i++)
    {
        Vector2 position = {
            particles->pPosX[i],
            particles->pPosY[i]
        };

        Vector2 velocity = {
            particles->pVelX[i],
            particles->pVelY[i]
        };

        const float mass =
            particles->pMasses[i];

        Vector2 force = {
            0.0f,
            0.0f
        };

        /*
         * Gravity.
         */

        force.y +=
            mass * gravityScale;

        /*
         * Viscosity.
         */

        const float viscosityForce =
            -6.0f *
            PI *
            totalViscosity *
            PARTICLE_RADIUS;

        force.x +=
            velocity.x *
            viscosityForce;

        force.y +=
            velocity.y *
            viscosityForce;

        /*
         * Attractors.
         */

        for (size_t j = 0;
             j < forces->attractCount;
             j++)
        {
            const float directionX =
                forces->attractPosX[j] -
                position.x;

            const float directionY =
                forces->attractPosY[j] -
                position.y;

            const float distanceSqr =
                directionX * directionX +
                directionY * directionY;

            if (distanceSqr < 1.0f)
            {
                continue;
            }

            const float distance =
                sqrtf(distanceSqr);

            const float normalizedX =
                directionX / distance;

            const float normalizedY =
                directionY / distance;

            const float strength =
                (mass * forces->attractMass[j]) /
                (distanceSqr + 10.0f);

            force.x +=
                normalizedX * strength;

            force.y +=
                normalizedY * strength;
        }

        /*
         * Repulsors.
         */

        for (size_t j = 0;
             j < forces->repulseCount;
             j++)
        {
            const float directionX =
                forces->repulsePosX[j] -
                position.x;

            const float directionY =
                forces->repulsePosY[j] -
                position.y;

            const float distanceSqr =
                directionX * directionX +
                directionY * directionY;

            if (distanceSqr < 1.0f)
            {
                continue;
            }

            const float distance =
                sqrtf(distanceSqr);

            const float normalizedX =
                directionX / distance;

            const float normalizedY =
                directionY / distance;

            const float strength =
                -(mass * forces->repulseMass[j]) /
                (distanceSqr + 10.0f);

            force.x +=
                normalizedX * strength;

            force.y +=
                normalizedY * strength;
        }

        /*
         * F = ma.
         */

        velocity.x +=
            force.x *
            deltaTime /
            mass;

        velocity.y +=
            force.y *
            deltaTime /
            mass;

        /*
         * Integrate position.
         */

        position.x +=
            velocity.x *
            deltaTime;

        position.y +=
            velocity.y *
            deltaTime;

        particles->pVelX[i] =
            velocity.x;

        particles->pVelY[i] =
            velocity.y;

        particles->pPrevPosX[i] =
            particles->pPosX[i];   

        particles->pPrevPosY[i] =
            particles->pPosY[i];

        particles->pPosX[i] =
            position.x;

        particles->pPosY[i] =
            position.y;
    }
}

static void UpdateParticlesMotion_(ParticleSystem *system, float deltaTime)
{
    ParticlePool *particles = system->particles_;
    // perform physics simulation updating particle attributes
    system->IntegrationFn(system, deltaTime);

    // Construct Spatial hash map of current particle positions.
    ClearHash(system->spatialHash);
    FillHash(system->spatialHash, system->particles_);

    // Generate self collision constraints
    size_t collisionCount = GenerateCollisionConstraints_(system);

    // Project constraints (solver)
    for (size_t i = 0; i < arrlenu(system->constraints_); i++)
    {
        const Constraint c = system->constraints_[i];
        c.ProjectFn(&c, system->particles_, deltaTime);
    }

    // Remove collision constraints
    // NOTE: Collision constraints must be added last because of removal strategy invariant.
    arrsetlen(system->constraints_, (arrlen(system->constraints_) - collisionCount));
    PASSERT((arrlen(system->constraints_) >= 0), LOG_ERROR, "");

    // Update velocities after constraint solver
    const float maxVelocity = 1000.0f; // Maximum velocity magnitude in pixels/second
    for (size_t i = 0; i < system->particles_->activeCount; i++)
    {
        
        particles->pVelX[i] =
            (particles->pPosX[i] - particles->pPrevPosX[i]) / deltaTime;

        particles->pVelY[i] =
            (particles->pPosY[i] - particles->pPrevPosY[i]) / deltaTime;
    }

    HandleBoundaryCollisions_(system);
}

ParticleSystem* ConstructParticleSystem(IntegratorType integrator, Vector4 boundary)
{
    ParticleSystem* system = (ParticleSystem*)malloc(sizeof(ParticleSystem));
    PASSERT(system, LOG_FATAL, "Failed to allocate particle pool");
    if(!system) { return NULL; }

    system->boundaryBox.left    = boundary.x;
    system->boundaryBox.right   = boundary.y;
    system->boundaryBox.bottom  = boundary.z;
    system->boundaryBox.top     = boundary.w;
    system->spatialHash = ConstructHash(2.0f * PARTICLE_RADIUS);

    system->emitter.position    = (Vector2){ 0 };
    system->emitter.radius      = EMITTER_RADIUS;
    
    system->constraints_    = NULL;
    arrsetcap(system->constraints_, MAX_PARTICLE_COUNT / 2);    // estimate likely maximum number of constraints
    system->forces_ = (ForcePool){ 0 };

    system->particles_ = ConstructParticlePool_();

    system->IntegrationFn = integrator == INTEGRATOR_VERLET ? IntegrateVerlet_ : IntegrateEuler_;

    return system;
}

void DestructParticleSystem(ParticleSystem *system)
{
    DestructHash(system->spatialHash);
    arrfree(system->constraints_);
    hmfree(system->forces_.addressMap);

    DestructParticlePool_(system->particles_);
    free(system);
}

void EmitParticles(ParticleSystem *system, const ParticleProps *props, uint32_t count)
{
    for (size_t c = 0; c < count; c++)
    {
        size_t i = system->particles_->activeCount;
        PASSERTRETURN(i < MAX_PARTICLE_COUNT, LOG_WARNING, "active particle count exceeds MAX_PARTICLE_COUNT");

        system->particles_->activeCount += 1;

        PASSERT((props->variance > -EPSILON && props->variance < (1.0 + EPSILON)),
            LOG_WARNING, "variance value outside valid range [0.0, 1.0]. Clamping value to valid range.");
        const float variance = Clamp(props->variance, 0.0f, 1.0f);

        system->particles_->pLifetimes[i]    = props->lifetime + (props->lifetime * (GetRandomValueF() * variance));
        system->particles_->pLifespans[i]    = 0;

        Vector2 position = Vector2Add(
        system->emitter.position,
        Vector2Scale(
            (Vector2){ GetRandomValueF(), GetRandomValueF() },
            system->emitter.radius * variance
            )
        );

        system->particles_->pPosX[i] = position.x;
        system->particles_->pPosY[i] = position.y;

        system->particles_->pPrevPosX[i] = position.x;
        system->particles_->pPrevPosY[i] = position.y;

        Vector2 velocity = Vector2Add(
            props->velocity,
            Vector2Scale(props->velocity, GetRandomValueF() * variance)
        );

        system->particles_->pVelX[i] = velocity.x;
        system->particles_->pVelY[i] = velocity.y;

        system->particles_->pMasses[i]       = props->mass;
    }
}

void UpdateParticles(ParticleSystem *system, float deltaTime)
{
    PASSERTRETURN((deltaTime > EPSILON), LOG_WARNING, "delta equal to zero. Skipping update step");

    UpdateParticlesLife_(system, deltaTime);
    // UpdateParticleAttributes_(system);

    const int substeps = PHYSICS_SUBSTEPS;
    const float deltaTimeSubstep = deltaTime / (float)substeps;
    for(size_t i = 0; i < substeps; i++)
    {
        UpdateParticlesMotion_(system, deltaTimeSubstep);
    }
}

void KillParticles(ParticleSystem *system, Vector2 position, float radius)
{
    QueryHashPoint(system->spatialHash, position, radius);
    for (size_t i = 0; i < arrlen(system->spatialHash->queryResults); i++) 
    {
        size_t pi = system->spatialHash->queryResults[i];

        Vector2 particlePosition = {
            system->particles_->pPosX[pi],
            system->particles_->pPosY[pi]
        };

        if (Vector2Distance(particlePosition, position) < radius)
        {
            KillParticle_(system->particles_, pi);
        }
    }
}

uint32_t AddForce(ParticleSystem *system, ForceType type)
{
    // maintain static uid int across all invocations
    static uint32_t uid = 0;

    ForcePool *forces = &system->forces_;

    if (forces->activeCount >= MAX_FORCES)
    {
        return UINT32_MAX;
    }

    const uint32_t forceId = uid++;

    ForceHandle handle = {
        .type = type,
        .index = 0,
    };

    switch (type)
    {
        case FORCE_GRAVITY:
        {
                const size_t index = forces->gravityCount;

                forces->gravityUids[index] = forceId;
                forces->gravityCount++;

                handle.index = index;
                break;
        }

        case FORCE_VISCOUS:
        {
                const size_t index = forces->viscosityCount;

                forces->viscosity[index] = AIR_VISCOSITY;
                forces->viscosityUids[index] = forceId;
                forces->viscosityCount++;

                handle.index = index;
                break;
        }

        case FORCE_ATTRACT:
        {
                const size_t index = forces->attractCount;

                forces->attractPosX[index] = 0.0f;
                forces->attractPosY[index] = 0.0f;
                forces->attractMass[index] = 0.0f;
                forces->attractUids[index] = forceId;

                forces->attractCount++;

                handle.index = index;
                break;
        }

        case FORCE_REPULSE:
        {
                const size_t index = forces->repulseCount;

                forces->repulsePosX[index] = 0.0f;
                forces->repulsePosY[index] = 0.0f;
                forces->repulseMass[index] = 0.0f;
                forces->repulseUids[index] = forceId;

                forces->repulseCount++;

                handle.index = index;
                break;
        }

        default:
            PASSERT(false, LOG_ERROR, "unknown force type.");
            return UINT32_MAX;
    }

    forces->activeCount++;

    hmput(
        forces->addressMap,
        forceId,
        handle
    );

    return forceId;
}

void SetForcePosition(
    ParticleSystem *system,
    uint32_t forceId,
    Vector2 position
)
{
    ForceHandle handle =
        hmget(system->forces_.addressMap, forceId);

    ForcePool *forces = &system->forces_;

    switch (handle.type)
    {
    case FORCE_ATTRACT:
        forces->attractPosX[handle.index] = position.x;
        forces->attractPosY[handle.index] = position.y;
        break;

    case FORCE_REPULSE:
        forces->repulsePosX[handle.index] = position.x;
        forces->repulsePosY[handle.index] = position.y;
        break;

    default:
        PASSERT(
            false,
            LOG_WARNING,
            "Force does not have a position."
        );
        break;
    }
}

void SetForceMass(
    ParticleSystem *system,
    uint32_t forceId,
    float mass
)
{
    ForceHandle handle =
        hmget(system->forces_.addressMap, forceId);

    ForcePool *forces = &system->forces_;

    switch (handle.type)
    {
    case FORCE_ATTRACT:
        forces->attractMass[handle.index] = mass;
        break;

    case FORCE_REPULSE:
        forces->repulseMass[handle.index] = mass;
        break;

    default:
        PASSERT(
            false,
            LOG_WARNING,
            "Force does not have mass."
        );
        break;
    }
}

void SetForceViscosity(
    ParticleSystem *system,
    uint32_t forceId,
    float viscosity
)
{
    ForceHandle handle =
        hmget(system->forces_.addressMap, forceId);

    ForcePool *forces = &system->forces_;

    PASSERT(
        handle.type == FORCE_VISCOUS,
        LOG_WARNING,
        "Force is not viscous."
    );

    if (handle.type != FORCE_VISCOUS)
    {
        return;
    }

    forces->viscosity[handle.index] = viscosity;
}

void InitParticleRender(const Shader *shader, float screenWidth, float screenHeight)
{
    shaderId = shader->id;
    quadVAO = rlLoadVertexArray();
    rlEnableVertexArray(quadVAO);
    quadVBO = rlLoadVertexBuffer(&quadVertices, sizeof(quadVertices), false);
    // aCoord
    rlEnableVertexAttribute(0);
    rlSetVertexAttribute(0, 2, RL_FLOAT, false, 4 * sizeof(float), 0);
    // aTexCoord
    rlEnableVertexAttribute(1);
    rlSetVertexAttribute(1, 2, RL_FLOAT, false, 4 * sizeof(float), 2 * sizeof(float));
    //  aPosition
    instancePositionVBO = rlLoadVertexBuffer(NULL, MAX_PARTICLE_COUNT * sizeof(Vector2), true);    // dynamic = true
    rlEnableVertexAttribute(2);
    rlSetVertexAttribute(2, 2, RL_FLOAT, false, 2 * sizeof(float), 0);
    rlSetVertexAttributeDivisor(2,1);

    rlDisableVertexBuffer();
    rlDisableVertexArray();

    float radius = PARTICLE_RADIUS;

    rlEnableShader(shader->id);
    rlSetUniform(GetShaderLocation(*shader, "uScreenWidth"), &screenWidth, RL_SHADER_UNIFORM_FLOAT, 1);
    rlSetUniform(GetShaderLocation(*shader, "uScreenHeight"), &screenHeight, RL_SHADER_UNIFORM_FLOAT, 1);
    rlSetUniform(GetShaderLocation(*shader, "uRadius"), &radius, RL_SHADER_UNIFORM_FLOAT, 1);
    rlDisableShader();
}

void CleanUpParticleRender()
{
    rlUnloadVertexArray(quadVAO);
    rlUnloadVertexBuffer(quadVBO);
    rlUnloadVertexBuffer(instancePositionVBO);

    quadVAO = 0;
    quadVBO = 0;
    instancePositionVBO = 0;
}

void DrawParticlesInstanced(const ParticleSystem *system)
{
    rlEnableShader(shaderId);
    rlEnableVertexArray(quadVAO);

    for (size_t i = 0; i < system->particles_->activeCount; i++)
    {
        instancePositions[i].x = system->particles_->pPosX[i];
        instancePositions[i].y = system->particles_->pPosY[i];
    }

    rlUpdateVertexBuffer(
        instancePositionVBO,
        instancePositions,
        system->particles_->activeCount * sizeof(Vector2),
        0
    );

    rlDrawVertexArrayInstanced(
        0,
        6,
        system->particles_->activeCount
    );

    rlDisableVertexArray();
    rlDisableShader();
}

void AddSelfCollisionConstraint(ParticleSystem *system, size_t i, size_t j)
{
    Constraint c = { 0 };
    c.type = CONSTRAINT_SELF_COLLISION;
    c.participants[0] = i;
    c.participants[1] = j;
    c.participantCount = 2;
    c.ProjectFn = ProjectSelfCollision;

    arrput(system->constraints_, c);
}

void AddDistanceConstraint(ParticleSystem *system, size_t i, size_t j)
{
    Constraint c = { 0 };
    c.type = CONSTRAINT_DISTANCE;
    c.participants[0] = i;
    c.participants[1] = j;
    c.participantCount = 2;
    c.ProjectFn = ProjectDistance;

    arrput(system->constraints_, c);
}
