#include "flamegpu/flamegpu.h"

#include <cmath>

FLAMEGPU_INIT_FUNCTION(create_agents) {
        int grid_size = FLAMEGPU -> environment.getProperty < int > ("GRID_SIZE");
        int countAgent[128][128] = {{0}};
        const unsigned int AGENT_COUNT = FLAMEGPU -> environment.getProperty < unsigned int > ("AGENT_COUNT");
        const float ENV_WIDTH = FLAMEGPU -> environment.getProperty <float > ("ENV_WIDTH");
        flamegpu::HostAgentAPI t_pop = FLAMEGPU -> agent("worm");
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            auto t = t_pop.newAgent();

            t.setVariable < float > ("x", FLAMEGPU -> random.uniform < float > (75.0f/128.0f,85.0f/128.0f)  *ENV_WIDTH);
            t.setVariable < float > ("y", FLAMEGPU -> random.uniform < float > (75.0f/128.0f,85.0f/128.0f) *ENV_WIDTH );
            t.setVariable <float > ("prev_direction_x", 0.0f);
            t.setVariable <float > ("prev_direction_y", 0.0f);
            // Calculate grid coordinatesfor the agent
            int grid_x =(int)  ((t.getVariable <float > ("x") / 20.0f) * 128);
            int grid_y =  (int)  ((t.getVariable <float > ("y")/20.0f) * 128);

            countAgent[grid_x][grid_y]++;
        }

        auto oxygen_grid = FLAMEGPU -> environment.getMacroProperty <float, 128, 128 > ("OXYGEN_GRID");
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                oxygen_grid[i][j] = 0.21f - (7.3 * pow(10, -4) / 0.65f) * (countAgent[i][j] / pow(2,(20.0/128.0)));
            }
        }
        auto attractant_grid = FLAMEGPU -> environment.getMacroProperty <float, 128, 128 > ("ATTRACTANT_GRID");
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                attractant_grid[i][j] = (countAgent[i][j] / pow(2,(20.0/128.0)))*0.01f*0.01f;
                if (i >= 59 && i <= 68 && j >= 59 && j <= 68) {
                    attractant_grid[i][j] += 1000.0f;
                }
            }
        }

        auto repellent_grid = FLAMEGPU -> environment.getMacroProperty <float, 128, 128 > ("REPELLENT_GRID");
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                repellent_grid[i][j] = (countAgent[i][j] / pow(2,(20.0/128.0)))*0.001f*0.001f;
            }
        }
}

FLAMEGPU_STEP_FUNCTION(update_grids) {
        int grid_size = FLAMEGPU -> environment.getProperty < int > ("GRID_SIZE");
        float env_size = FLAMEGPU -> environment.getProperty < float > ("ENV_WIDTH");
        int attractant = FLAMEGPU -> environment.getProperty < int > ("ATTRACTANT");
        int repellent = FLAMEGPU -> environment.getProperty < int > ("REPELLENT");
        int oxygen = FLAMEGPU -> environment.getProperty < int > ("OXYGEN");
        int both = FLAMEGPU -> environment.getProperty < int > ("PHEROMONES");
        if (both != 0){
            FLAMEGPU ->environment.setProperty("REPELLENT",1);
            FLAMEGPU ->environment.setProperty("ATTRACTANT",1);
        }
        auto density_grid = FLAMEGPU->environment.getMacroProperty<int,128,128>("DENSITY_GRID");
        auto oxygen_grid = FLAMEGPU->environment.getMacroProperty<float, 128, 128>("OXYGEN_GRID");
        float new_oxygen_grid[128][128];


        auto attractant_grid = FLAMEGPU->environment.getMacroProperty<float, 128, 128>("ATTRACTANT_GRID");
        float new_attractant_grid[128][128];

        auto repellent_grid = FLAMEGPU->environment.getMacroProperty<float, 128, 128>("REPELLENT_GRID");
        float new_repellent_grid[128][128];

        float h = (20.0f/128.0)*(20.0f/128.0);
        int countAgent[128][128] = {{0}};



        float speed = 0.0;
        float agent_x = 0.0;
        float agent_y = 0.0;
        for (int i=0; i <128;i++){
            for (int j=0;j<128;j++){
                density_grid[i][j]=0;
            }
        }
        const std::string outputFileNameCoordinates = "../countAgent/" + std::to_string(FLAMEGPU->getStepCounter()) + ".csv";
        for (const auto & agent: FLAMEGPU -> agent("worm").getPopulationData()) {
            // Access agent position variables
            agent_x = agent.getVariable <float > ("x");
            agent_y = agent.getVariable <float > ("y");
            // Calculate grid coordinatesfor the agent
            int grid_x =(int)  ((agent_x / 20.0f) * 128);
              int grid_y =  (int)  ((agent_y/20.0f)* 128);
            if (grid_x > 128){
                grid_x=0;
            }
            if (grid_y >128){
                grid_y=0;
            }
            density_grid[grid_x][grid_y]++;

        }
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {

                int previous_x = i - 1;
                int previous_y = j - 1;
                int next_x = i + 1;
                int next_y = j + 1;
                if (previous_x < 0) {
                    previous_x = 127;
                }
                if (previous_y < 0) {
                    previous_y = 127;
                }
                if (next_x >= 128) {
                    next_x = 0;
                }
                if (next_y >= 128) {
                    next_y = 0;
                }

                if (oxygen != 0) {

                    float oxygen_consumption = FLAMEGPU->environment.getProperty<float>("OXYGEN_CONSUMPTION");
                    float oxygen_global = FLAMEGPU->environment.getProperty<float>("OXYGEN_GLOBAL");
                    float oxygen_diffusion = FLAMEGPU->environment.getProperty < float > ("OXYGEN_DIFFUSION");
                    float laplacianO = (oxygen_grid[next_x][j] + oxygen_grid[previous_x][j] + oxygen_grid[i][next_y] + oxygen_grid[i][previous_y] - 4 * oxygen_grid[i][j]) / h;
                    new_oxygen_grid[i][j] = 0.65f * (oxygen_global - oxygen_grid[i][j]) - ((density_grid[i][j]) / h) * oxygen_consumption + oxygen_diffusion * laplacianO;

                }
                if (attractant != 0) {
                    float attractant_creation =  FLAMEGPU->environment.getProperty<float>("ATTRACTANT_CREATION");
                    float laplacianO = (attractant_grid[next_x][j] + attractant_grid[previous_x][j] + attractant_grid[i][next_y] +attractant_grid[i][previous_y] - 4 * attractant_grid[i][j]) / h;
                    new_attractant_grid[i][j] = -0.01f*attractant_grid[i][j] + 0.000001f * laplacianO * attractant_grid[i][j] +attractant_creation * ((density_grid[i][j]) / h);

                }
                if (repellent != 0) {
                    float repellent_creation =  FLAMEGPU->environment.getProperty<float>("REPELLENT_CREATION");
                    float laplacianO =(repellent_grid[next_x][j] + repellent_grid[previous_x][j] + repellent_grid[i][next_y] +repellent_grid[i][previous_y] - 4 * repellent_grid[i][j]) / h;
                    new_repellent_grid[i][j] = -0.001f*repellent_grid[i][j] + 0.00001f * laplacianO * repellent_grid[i][j] +repellent_creation * ((density_grid[i][j]) / h);
                }
            }
        }
        if (repellent !=0){
            for (int i = 0; i < 128; ++i) {
                for (int j = 0; j < 128; ++j) {
                    repellent_grid[i][j] += new_repellent_grid[i][j];
                }
            }

        }
        if (attractant != 0){
            for (int i = 0; i < 128; ++i) {
                for (int j = 0; j < 128; ++j) {
                    attractant_grid[i][j] += new_attractant_grid[i][j];
                    if (i >= 59 && i <= 68 && j >= 59 && j <= 68)  {
                        attractant_grid[i][j] += 1.0f;
                    }
                }
            }
        }
        if (oxygen !=0){
            for (int i = 0; i < 128; ++i) {
                for (int j = 0; j < 128; ++j) {
                    oxygen_grid[i][j] += new_oxygen_grid[i][j];

                }
            }

        }
}

FLAMEGPU_AGENT_FUNCTION(output_message, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
  FLAMEGPU -> message_out.setVariable < int > ("id", FLAMEGPU -> getID());
  FLAMEGPU -> message_out.setLocation(FLAMEGPU -> getVariable < float > ("x"), FLAMEGPU -> getVariable < float > ("y"));
  return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(input_message, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
  const flamegpu::id_t ID = FLAMEGPU -> getID();
  int grid_size = FLAMEGPU -> environment.getProperty < int > ("GRID_SIZE");
  int attractant = FLAMEGPU -> environment.getProperty < int > ("ATTRACTANT");
  int repellent = FLAMEGPU -> environment.getProperty < int > ("REPELLENT");
  int oxygen = FLAMEGPU -> environment.getProperty < int > ("OXYGEN");
  int external_attractant = FLAMEGPU -> environment.getProperty < int > ("EXTERNAL_ATTRACTANT");

  const float x1 = FLAMEGPU -> getVariable < float > ("x");
  const float y1 = FLAMEGPU -> getVariable < float > ("y");

  int grid_x =(int)  (((x1 / 20.0f) * 128));
  int grid_y =  (int)  (((y1 / 20.0f) * 128));

  float persistence_factor = FLAMEGPU -> environment.getProperty < float > ("PERSISTENCE_FACTOR");
  float beta_attractant = FLAMEGPU -> environment.getProperty < float > ("BETA_ATTRACTANT");
  float beta_repellent = FLAMEGPU -> environment.getProperty < float > ("BETA_REPELLENT");
  float alpha_attractant = FLAMEGPU -> environment.getProperty < float > ("ALPHA_ATTRACTANT");
  float alpha_repellent = FLAMEGPU -> environment.getProperty < float > ("ALPHA_REPELLENT");

  auto density_grid = FLAMEGPU -> environment.getMacroProperty < int, 128, 128 > ("DENSITY_GRID");
  auto oxygen_grid = FLAMEGPU -> environment.getMacroProperty < float, 128, 128 > ("OXYGEN_GRID");
  auto attractant_grid = FLAMEGPU -> environment.getMacroProperty < float, 128, 128 > ("ATTRACTANT_GRID");
  auto repellent_grid = FLAMEGPU -> environment.getMacroProperty < float, 128, 128 > ("REPELLENT_GRID");

  float prev_direction_x = FLAMEGPU -> getVariable < float > ("prev_direction_x");
  float prev_direction_y = FLAMEGPU -> getVariable < float > ("prev_direction_y");


  float random_angle = FLAMEGPU -> random.uniform < float > (0.0f, 2 * 3.14159f);
  float new_direction_x = cosf(random_angle);
  float new_direction_y = sinf(random_angle);


  float repellent_direction_x = 0.0f;
  float repellent_direction_y = 0.0f;
  float attractant_direction_x = 0.0f;
  float attractant_direction_y = 0.0f;


  float best = attractant_grid[grid_x][grid_y];
  int N = FLAMEGPU -> environment.getProperty < int > ("SENSING_RANGE");

  for (int i = (int) - N / 2; i <= (int) N / 2; i++) {
    for (int j = (int) - N / 2; j <= (int) N / 2; j++) {

        int neighbor_x = grid_x + i;
        int neighbor_y = grid_y + j;

        if (neighbor_x < 0)
            neighbor_x += 128;
        else if (neighbor_x >= 128)
            neighbor_x -= 128;

        if (neighbor_y < 0)
            neighbor_y += 128;
        else if (neighbor_y >= 128)
            neighbor_y -= 128;


      if (attractant_grid[neighbor_x][neighbor_y] > best && density_grid[neighbor_x][neighbor_y] <= 4) {
        best = attractant_grid[neighbor_x][neighbor_y];
        float dx = neighbor_x - grid_x;
        float dy = neighbor_y - grid_y;

        float angle = atan2f(dy, dx); // Corrected angle calculation
        angle += FLAMEGPU -> random.uniform < float > (-0.5f, 0.5f);
        attractant_direction_x = cosf(angle);
        attractant_direction_y = sinf(angle);
      }
    }
  }



  float best_r = repellent_grid[grid_x][grid_y];
  int N1 = FLAMEGPU -> environment.getProperty < int > ("SENSING_RANGE") * 10;

  float h = (20.0f/128.0)*(20.0f/128.0);

  for (int i = -N1 / 2; i <= N1 / 2; i++) {
    for (int j = -N1 / 2; j <= N1 / 2; j++) {
      // Calculate indices of neighboring cell with wrap-around
        int neighbor_x = grid_x + i;
        int neighbor_y = grid_y + j;

        // Wrap around if out of bounds
        if (neighbor_x < 0)
            neighbor_x += 128;
        else if (neighbor_x >= 128)
            neighbor_x -= 128;

        if (neighbor_y < 0)
            neighbor_y += 128;
        else if (neighbor_y >= 128)
            neighbor_y -= 128;
      if (repellent_grid[neighbor_x][neighbor_y] > best_r && density_grid[neighbor_x][neighbor_y] <= 4) {
        best_r = repellent_grid[neighbor_x][neighbor_y];
        float dx = neighbor_x - grid_x;
        float dy = neighbor_y - grid_y;
        float angle = atan2f(dy, dx);
        angle += FLAMEGPU -> random.uniform < float > (-0.5f, 0.5f);
        repellent_direction_x = cosf(angle + 3.14);
        repellent_direction_y = sinf(angle + 3.14);
      }
    }
  }



  float sensed_oxygen = (float) oxygen_grid[grid_x][grid_y];
  float sensed_phero = (float) attractant_grid[grid_x][grid_y];
  float sensed_repellent = (float) repellent_grid[grid_x][grid_y];

  float speed = 0.0015f;
  if (oxygen != 0) {
    speed = (18.5f * sensed_oxygen * sensed_oxygen - 0.398f * sensed_oxygen + 0.0225f)/100;
  }
  if (attractant != 0 && repellent != 0 && oxygen ==0) {
      if (sensed_phero > 0.02f) {
          float V_U = beta_attractant * log10(alpha_attractant + sensed_phero);
          float V_Ur = -beta_repellent * log10(alpha_repellent + sensed_repellent);
          speed = V_Ur + V_U;
          //printf("%f , " ,speed);
          if (speed < 0) {
              if (sensed_repellent != best_r) {
                  new_direction_x = cosf(random_angle) + (0.01f * best_r) * repellent_direction_x;
                  new_direction_y = sinf(random_angle) + (0.01f * best_r) * repellent_direction_y;
              }
          } else {
              if (sensed_phero != best) {
                  new_direction_x = cosf(random_angle) + (0.01f * best) * attractant_direction_x;
                  new_direction_y = sinf(random_angle) + (0.01f * best) * attractant_direction_y;
              }
          }
          speed = abs(speed);
      }
  }
if (attractant != 0 && repellent != 0 && oxygen != 0 ) {
    float V_U = beta_attractant * log10(alpha_attractant + sensed_phero);
    float V_Ur = -beta_repellent * log10(alpha_repellent + sensed_repellent);
    float h= FLAMEGPU -> environment.getProperty < float > ("H");
    speed = h*((18.5f * sensed_oxygen * sensed_oxygen - 0.398f * sensed_oxygen + 0.0225f )/100)+ ((V_Ur + V_U))*(1.0f-h);
    if (speed < 0) {
      if (sensed_repellent != best_r) {
        new_direction_x = cosf(random_angle) + (0.01f * (1.0f-h) * best_r) * repellent_direction_x;
        new_direction_y = sinf(random_angle) + (0.01f*(1.0f-h)*  best_r) * repellent_direction_y;
      }
    } else {
      if (sensed_phero != best) {
        new_direction_x = cosf(random_angle) + (0.01f*(1.0f-h) * best) * attractant_direction_x;
        new_direction_y = sinf(random_angle) + (0.01f*(1.0f-h) * best) * attractant_direction_y;
      }
    }
  }

  float fx = persistence_factor * prev_direction_x + (1.0f - persistence_factor) * new_direction_x;
  float fy = persistence_factor * prev_direction_y + (1.0f - persistence_factor) * new_direction_y;

  float len = sqrt(fx * fx + fy * fy);
  fx /= len;
  fy /= len;

  float new_x = x1 + fx*speed;
  float new_y = y1 + fy*speed;

  // Update agent positions and previous direction
  float width = FLAMEGPU -> environment.getProperty < float > ("ENV_WIDTH");
  if (new_x < 0) {
    new_x = width + fx * speed;
  } else if (new_x >= width) {
    new_x = fx * speed;
  }
  if (new_y < 0) {
    new_y = width + fy * speed;
  } else if (new_y >= width) {
    new_y =  fy * speed;
  }
  FLAMEGPU -> setVariable < float > ("x", new_x);
  FLAMEGPU -> setVariable < float > ("y", new_y);
  FLAMEGPU -> setVariable < float > ("prev_direction_x", fx);
  FLAMEGPU -> setVariable < float > ("prev_direction_y", fy);

  return flamegpu::ALIVE;
}

int main(int argc, const char ** argv) {

    const float ENV_WIDTH = 20;
    const unsigned int AGENT_COUNT = 40 * ENV_WIDTH * ENV_WIDTH;


    flamegpu::ModelDescription model("V2.1");

    {
        flamegpu::MessageSpatial2D::Description message = model.newMessage <flamegpu::MessageSpatial2D > ("location");
        message.setMin(0, 0);
        message.setMax(ENV_WIDTH, ENV_WIDTH);
        message.setRadius(0.5f);
        message.newVariable <flamegpu::id_t > ("id");
    }

    flamegpu::AgentDescription agent = model.newAgent("worm");
    agent.newVariable <float > ("x");
    agent.newVariable <float > ("y");
    agent.newVariable <float > ("prev_direction_x");
    agent.newVariable <float > ("prev_direction_y");

    flamegpu::AgentFunctionDescription out_fn = agent.newFunction("output_message", output_message);
    out_fn.setMessageOutput("location");
    flamegpu::AgentFunctionDescription in_fn = agent.newFunction("input_message", input_message);
    in_fn.setMessageInput("location");
    {
        flamegpu::EnvironmentDescription env = model.Environment();
        env.newProperty < unsigned int > ("AGENT_COUNT", AGENT_COUNT);
        env.newProperty <float > ("ENV_WIDTH", ENV_WIDTH);
        env.newProperty <int > ("GRID_SIZE", 128);
        env.newProperty <float > ("PERSISTENCE_FACTOR", 0.8f);
        env.newMacroProperty <float, 128, 128 > ("OXYGEN_GRID");
        env.newMacroProperty <float, 128, 128 > ("ATTRACTANT_GRID");
        env.newMacroProperty < int, 128, 128 > ("DENSITY_GRID");
        env.newMacroProperty <float, 128, 128 > ("REPELLENT_GRID");
        env.newProperty <float > ("OXYGEN_CONSUMPTION", 0.0007f);
        env.newProperty <float > ("OXYGEN_GLOBAL", 0.21f);
        env.newProperty <float > ("OXYGEN_DIFFUSION", 0.002f);
        env.newProperty < int > ("SENSING_RANGE", 2);
        //strenght of attraction
        env.newProperty<float>("BETA_ATTRACTANT",0.001111f);
        env.newProperty<float>("BETA_REPELLENT",0.001111f);
        env.newProperty<float>("ATTRACTANT_CREATION",0.01f);
//concentration scale
        env.newProperty<float>("ALPHA_ATTRACTANT",15);
        env.newProperty<float>("ALPHA_REPELLENT",15);
        env.newProperty<float>("REPELLENT_CREATION",0.001f);
        env.newProperty < float > ("H", 0.5f);

        env.newProperty < int > ("OXYGEN", 0);
        env.newProperty < int > ("ATTRACTANT", 0);
        env.newProperty < int > ("REPELLENT", 0);
        env.newProperty < int > ("PHEROMONES", 1);
        env.newProperty < int > ("EXTERNAL_ATTRACTANT", 0);

    } {
        in_fn.dependsOn(out_fn);
        model.addExecutionRoot(out_fn);
        model.generateLayers();
    }

    model.addInitFunction(create_agents);
    model.addStepFunction(update_grids);


    flamegpu::CUDASimulation cuda_model(model, argc, argv);





#ifdef FLAMEGPU_VISUALISATION
  flamegpu::visualiser::ModelVis m_vis = cuda_model.getVisualisation();
  flamegpu::visualiser::PanelVis ui = m_vis.newUIPanel("Settings");

  ui.newEnvironmentPropertySlider <float > ("PERSISTENCE_FACTOR", 0.0f, 1.0f);
  ui.newSeparator();
  ui.newEnvironmentPropertyToggle < int > ("OXYGEN");
  ui.newEnvironmentPropertySlider <float > ("OXYGEN_CONSUMPTION", 0.0f, 0.001f);
  ui.newEnvironmentPropertySlider <float > ("OXYGEN_GLOBAL", 0.0f, 0.21f);
  ui.newEnvironmentPropertySlider <float > ("OXYGEN_DIFFUSION", 0.0f, 0.005f);
  ui.newSeparator();
  ui.newEnvironmentPropertyToggle < int > ("PHEROMONES");
  ui.newEnvironmentPropertySlider <float > ("BETA_ATTRACTANT", 0.0001f, 0.005f);
  ui.newEnvironmentPropertySlider <float > ("ALPHA_ATTRACTANT", 10.0f, 20.0f);
  ui.newEnvironmentPropertySlider <float > ("ATTRACTANT_CREATION", 0.0f, 0.1f);
  ui.newEnvironmentPropertySlider <float > ("BETA_REPELLENT", 0.0001f, 0.005f);
  ui.newEnvironmentPropertySlider <float > ("ALPHA_REPELLENT", 10.0f, 20.0f);
  ui.newEnvironmentPropertySlider <float > ("REPELLENT_CREATION", 0.0f, 0.01f);
  ui.newEnvironmentPropertySlider < int > ("SENSING_RANGE", 0, 20);

  ui.newSeparator();
  ui.newEnvironmentPropertyToggle < int > ("EXTERNAL_ATTRACTANT");
  const float INIT_CAM = ENV_WIDTH / 2.0f;
  m_vis.setInitialCameraTarget(INIT_CAM, INIT_CAM, 0);
  m_vis.setInitialCameraLocation(INIT_CAM, INIT_CAM, ENV_WIDTH);
  m_vis.setCameraSpeed(0.01f);
  m_vis.setSimulationSpeed(24);

  flamegpu::visualiser::AgentVis worm_agt = m_vis.addAgent("worm");

  worm_agt.setModel(flamegpu::visualiser::Stock::Models::ICOSPHERE);

  worm_agt.setColor(flamegpu::visualiser::Color{"#ffffff"});
  worm_agt.setModelScale(0.03f);

  flamegpu::visualiser::LineVis pen = m_vis.newPolylineSketch(1, 1, 1, 0.2f);
  pen.addVertex(0, 0, 0);
  pen.addVertex(0, ENV_WIDTH, 0);
  pen.addVertex(ENV_WIDTH, ENV_WIDTH, 0);
  pen.addVertex(ENV_WIDTH, 0, 0);
  pen.addVertex(0, 0, 0);
// Calculate the center of the environment
float centerX = ENV_WIDTH / 2.0f;
float centerY = ENV_WIDTH / 2.0f;

// Define the size of the square
float squareSize = (ENV_WIDTH / 128.0f)*10;

// Create a new polyline sketch
flamegpu::visualiser::LineVis pen2 = m_vis.newPolylineSketch(1.0f, 0.0f, 0.0f, 0.8f);

// Add vertices to draw the square
pen2.addVertex(centerX - squareSize / 2, centerY - squareSize / 2, 0); // Top-left corner
pen2.addVertex(centerX + squareSize / 2, centerY - squareSize / 2, 0); // Top-right corner
pen2.addVertex(centerX + squareSize / 2, centerY + squareSize / 2, 0); // Bottom-right corner
pen2.addVertex(centerX - squareSize / 2, centerY + squareSize / 2, 0); // Bottom-left corner
pen2.addVertex(centerX - squareSize / 2, centerY - squareSize / 2, 0); // Back to top-left to complete the square

// Close the square
pen2.addVertex(centerX - squareSize / 2, centerY - squareSize / 2, 0);
  m_vis.activate();
#endif
    cuda_model.simulate();
    cuda_model.exportData("/home/aymeric/CLionProjects/FLAMEGPU2-babots/babots/worm_aggregation/src/log.json");

#ifdef FLAMEGPU_VISUALISATION
  m_vis.join();
#endif
}