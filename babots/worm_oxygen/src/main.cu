#include "flamegpu/flamegpu.h"
#include <cmath>





FLAMEGPU_AGENT_FUNCTION(output_message, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setLocation(FLAMEGPU->getVariable<float>("x"),
                                        FLAMEGPU->getVariable<float>("y"));
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(input_message, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    const flamegpu::id_t ID = FLAMEGPU->getID();
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");


    // Persistence factor (adjust how persistent the direction is 1=very 0=none)
    float persistence_factor = FLAMEGPU -> environment.getProperty < float > ("PERSISTENCE_FACTOR");

    // Retrieve the previous direction of the agent
    float prev_direction_x = FLAMEGPU->getVariable<float>("prev_direction_x");
    float prev_direction_y = FLAMEGPU->getVariable<float>("prev_direction_y");


    // Generate a new random angle  (only half turn, *2pi for 360d)
    float random_angle = FLAMEGPU->random.uniform<float>(0.0f,  2*3.14159f);

    // Calculate movement components based on the new and previous directions
    float new_direction_x = cosf(random_angle);
    float new_direction_y = sinf(random_angle);

    // Apply persistence to smoothen the movement
    float fx = persistence_factor * prev_direction_x + (1.0f - persistence_factor) * new_direction_x;
    float fy = persistence_factor * prev_direction_y + (1.0f - persistence_factor) * new_direction_y;

    //compute speed
    float sensing_range = FLAMEGPU -> environment.getProperty < float > ("SENSING_RANGE");
    float oxygen_consumption = FLAMEGPU -> environment.getProperty < float > ("OXYGEN_CONSUMPTION");
    float oxygen_global = FLAMEGPU -> environment.getProperty < float > ("OXYGEN_GLOBAL_LEVEL");
    int count = 0;


    for (const auto &message : FLAMEGPU->message_in(x1, y1)) {
        if (message.getVariable<flamegpu::id_t>("id") != ID) {
            const float x2 = message.getVariable<float>("x");
            const float y2 = message.getVariable<float>("y");

            float x21 = x2 - x1;
            float y21 = y2 - y1;
            const float separation = sqrt(x21*x21 + y21*y21);
            if (separation < sensing_range && separation > 0.0f) {
                count++;
            }


        }
    }

    float sensed_oxygen = oxygen_global - count * oxygen_consumption;


    //float x = ((sensed_oxygen - 14.0f) / (oxygen_global - 14.0f));
    //if (x < 0){
        //x = speed/10.0f;
    //}

    float speed = 18.9f * sensed_oxygen * sensed_oxygen - 3.98 * sensed_oxygen + 0.225;
    float new_x = x1 + fx*speed;
    float new_y = y1 + fy*speed;
    float width = FLAMEGPU -> environment.getProperty < float > ("ENV_WIDTH");
    if (new_x <= 0){
        new_x = width + fx*speed;
    } else if (new_x >= width){
        new_x = 0 + fx*speed;
    }
    if (new_y <= 0){
        new_y = width + fy*speed;
    } else if (new_y >= width){
        new_y = 0 + fy*speed;
    }


    // Update agent positions and previous direction
    FLAMEGPU->setVariable<float>("x", new_x);
    FLAMEGPU->setVariable<float>("y", new_y);
    FLAMEGPU->setVariable<float>("oxygen_sensing", sensed_oxygen);
    FLAMEGPU->setVariable<float>("prev_direction_x", fx);
    FLAMEGPU->setVariable<float>("prev_direction_y", fy);


    return flamegpu::ALIVE;
}
FLAMEGPU_INIT_FUNCTION(create_agents) {
        // Fetch the desired agent count and environment width
        const unsigned int AGENT_COUNT = FLAMEGPU -> environment.getProperty < unsigned int > ("AGENT_COUNT");
        const float ENV_WIDTH = FLAMEGPU -> environment.getProperty < float > ("ENV_WIDTH");
        // Create agents
        flamegpu::HostAgentAPI t_pop = FLAMEGPU -> agent("worm");
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            auto t = t_pop.newAgent();
            t.setVariable < float > ("x", FLAMEGPU -> random.uniform < float > () * ENV_WIDTH );
            t.setVariable < float > ("y", FLAMEGPU -> random.uniform < float > () * ENV_WIDTH );
            t.setVariable < float > ("prev_direction_x", 0.0f);
            t.setVariable < float > ("prev_direction_y", 0.0f);
            t.setVariable < float > ("oxygen_sensing", 0.21f);
        }

}

FLAMEGPU_STEP_FUNCTION(Validation){
    int p = FLAMEGPU -> environment.getProperty < int > ("PRINT");
    if (p == 100){
            // Set PRINT property and retrieve environment parameters
            FLAMEGPU->environment.setProperty("PRINT", 1);
            const float ENV_WIDTH = FLAMEGPU->environment.getProperty<float>("ENV_WIDTH");
            const int GRID_SIZE = 128;
            const float CELL_SIZE = ENV_WIDTH / GRID_SIZE;

// Vector to store square densities and countAgent array initialization
            std::vector<float> squareDensities;
            int countAgent[GRID_SIZE][GRID_SIZE] = {{0}};

// Count agents in respective cells
            for (const auto &agent : FLAMEGPU->agent("worm").getPopulationData()) {
                // Access agent position variables
                float agent_x = agent.getVariable<float>("x");
                float agent_y = agent.getVariable<float>("y");

                // Calculate grid coordinates for the agent
                int grid_x = static_cast<int>(std::floor(agent_x / CELL_SIZE));
                int grid_y = static_cast<int>(std::floor(agent_y / CELL_SIZE));

                // Increment the count of agents in the corresponding cell
                countAgent[grid_x][grid_y]++;
            }

// Calculate square densities
            for (int i = 0; i < GRID_SIZE; ++i) {
                for (int j = 0; j < GRID_SIZE; ++j) {
                    int agentsInSquare = countAgent[i][j];
                    // Calculate density by dividing the number of agents by the area of the square
                    float density = static_cast<float>(agentsInSquare) / (CELL_SIZE * CELL_SIZE);
                    squareDensities.push_back(density);
                }
            }

// Calculate mean and normalize densities to probabilities
            float maxDensity = *std::max_element(squareDensities.begin(), squareDensities.end());

            for (float &density : squareDensities) {
                density /= maxDensity; // Normalize densities to probabilities (values between 0 and 1)
            }
            float sum = 0.0f;
            for (float density : squareDensities) {
                sum += density;
            }
            float mean = sum / squareDensities.size();

// Calculate variance and standard deviation
            float variance = 0.0f;
            for (const auto &value : squareDensities) {
                variance += std::pow(value - mean, 2);
            }
            variance /= squareDensities.size();
            float std_dev = sqrt(variance);

// Normalize standard deviation using Z-score normalization
            std_dev = (std_dev - mean) / std_dev;
            std_dev = std::abs(std_dev);

// Calculate entropy of density distribution
            float entropy = 0.0f;
            for (float density : squareDensities) {
                if (density > 0.0f) { // Ensure probability is positive to avoid log(0)
                    entropy -= density * log2f(density); // Use log base 2 for entropy calculation
                }
            }

// Print results
            int currentStep = FLAMEGPU->getStepCounter();
            printf("Step %d: Entropy of density distribution: %f\n", currentStep, entropy);
            printf("Step %d: Standard deviation of density: %f\n", currentStep, std_dev);

        } else {
        p++;
        FLAMEGPU->environment.setProperty("PRINT",p);
    }

}

int main(int argc, const char ** argv) {
    // Define some useful constants
    const float ENV_WIDTH = 20;
    const unsigned int AGENT_COUNT = 12000 * ENV_WIDTH;


    // Define the FLAME GPU model
    flamegpu::ModelDescription model("V1.2");

    { // (optional local scope block for cleaner grouping)
        // Define a message of type MessageSpatial2D named location
        flamegpu::MessageSpatial2D::Description message = model.newMessage < flamegpu::MessageSpatial2D > ("location");
        // Configure the message list
        message.setMin(0, 0);
        message.setMax(ENV_WIDTH, ENV_WIDTH);
        message.setRadius(0.5f);
        // Add extra variables to the message
        // X Y (Z) are implicit for spatial messages
        message.newVariable < flamegpu::id_t > ("id");
    }

    // Define an agent named worm
    flamegpu::AgentDescription agent = model.newAgent("worm");
    // Assign the agent some variables (ID is implicit to agents, so we don't define it ourselves)
    agent.newVariable < float > ("x");
    agent.newVariable < float > ("y");
    agent.newVariable < float > ("prev_direction_x");
    agent.newVariable < float > ("prev_direction_y");
    agent.newVariable < float > ("oxygen_sensing");

    flamegpu::AgentFunctionDescription out_fn = agent.newFunction("output_message", output_message);
    out_fn.setMessageOutput("location");
    flamegpu::AgentFunctionDescription in_fn = agent.newFunction("input_message", input_message);
    in_fn.setMessageInput("location");
    { // (optional local scope block for cleaner grouping)
        // Define environment properties
        flamegpu::EnvironmentDescription env = model.Environment();
        env.newProperty < unsigned int > ("AGENT_COUNT", AGENT_COUNT);
        env.newProperty < float > ("ENV_WIDTH", ENV_WIDTH);
        env.newProperty < float > ("SENSING_RANGE", 0.2f);
        env.newProperty < float > ("OXYGEN_CONSUMPTION", 0.0007f);
        env.newProperty < float > ("OXYGEN_GLOBAL_LEVEL", 0.21f);
        env.newProperty < float > ("PERSISTENCE_FACTOR", 0.95f);
        env.newProperty < int > ("PRINT", 0);

    }

    {   // (optional local scope block for cleaner grouping)
        // Dependency specification
        // Message input depends on output
        in_fn.dependsOn(out_fn);
        // Output is the root of our graph
        model.addExecutionRoot(out_fn);
        //deactivate here
        model.addStepFunction(Validation);
        model.generateLayers();

    }

    model.addInitFunction(create_agents);



    // Specify the desired StepLoggingConfig
    flamegpu::StepLoggingConfig step_log_cfg(model);
    // Log every 100 step
    step_log_cfg.setFrequency(1000);
    // Include the mean of the "point" agent population's variable 'drift'

    // Create the simulation
    flamegpu::CUDASimulation cuda_model(model, argc, argv);

    // Attach the logging config
    cuda_model.setStepLog(step_log_cfg);

    // Only compile this block if being built with visualisation support
#ifdef FLAMEGPU_VISUALISATION
    // Create visualisation
  flamegpu::visualiser::ModelVis m_vis = cuda_model.getVisualisation();


    flamegpu::visualiser::PanelVis ui = m_vis.newUIPanel("Settings");
    ui.newEnvironmentPropertySlider<float>("SENSING_RANGE", 0.0f, 0.5f);
    ui.newEnvironmentPropertySlider<float>("OXYGEN_CONSUMPTION", 0.0f, 0.01f);
    ui.newEnvironmentPropertySlider<float>("OXYGEN_GLOBAL_LEVEL", 0.0f, 0.3f);
    ui.newEnvironmentPropertySlider<float>("PERSISTENCE_FACTOR", 0.0f,1.0f);

  // Set the initial camera location and speed
  const float INIT_CAM = ENV_WIDTH / 2.0f;
  m_vis.setInitialCameraTarget(INIT_CAM, INIT_CAM, 0);
  m_vis.setInitialCameraLocation(INIT_CAM, INIT_CAM, ENV_WIDTH);
  m_vis.setCameraSpeed(0.01f);
  m_vis.setSimulationSpeed(24);
  // Add worm agents to the visualisation
  flamegpu::visualiser::AgentVis worm_agt = m_vis.addAgent("worm");

  // Location variables have names "x" and "y" so will be used by default
  worm_agt.setModel(flamegpu::visualiser::Stock::Models::ICOSPHERE);
  //head is 3 micrometer
  worm_agt.setModelScale(0.03f);


     // Mark the environment bounds
    flamegpu::visualiser::LineVis pen = m_vis.newPolylineSketch(1, 1, 1, 0.2f);
    pen.addVertex(0, 0, 0);
    pen.addVertex(0, ENV_WIDTH, 0);
    pen.addVertex(ENV_WIDTH, ENV_WIDTH, 0);
    pen.addVertex(ENV_WIDTH, 0, 0);
    pen.addVertex(0, 0, 0);
  // Open the visualiser window
  m_vis.activate();
#endif

    // Run the simulation
    cuda_model.simulate();

#ifdef FLAMEGPU_VISUALISATION
    // Keep the visualisation window active after the simulation has completed
  m_vis.join();
#endif
}