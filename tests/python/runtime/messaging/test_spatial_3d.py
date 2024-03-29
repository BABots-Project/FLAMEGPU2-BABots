import pytest
from unittest import TestCase
from pyflamegpu import *
import random as rand

AGENT_COUNT = 2049

out_mandatory3D = """
FLAMEGPU_AGENT_FUNCTION(out_mandatory3D, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"),
        FLAMEGPU->getVariable<float>("z"));
    return flamegpu::ALIVE;
}
"""

out_optional3D = """
FLAMEGPU_AGENT_FUNCTION(out_optional3D, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    if (FLAMEGPU->getVariable<int>("do_output")) {
        FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
        FLAMEGPU->message_out.setLocation(
            FLAMEGPU->getVariable<float>("x"),
            FLAMEGPU->getVariable<float>("y"),
            FLAMEGPU->getVariable<float>("z"));
    }
    return flamegpu::ALIVE;
}
"""

in3D = """
FLAMEGPU_AGENT_FUNCTION(in3D, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");
    const float z1 = FLAMEGPU->getVariable<float>("z");
    unsigned int count = 0;
    unsigned int badCount = 0;
     int myBin[3] = {
         static_cast<int>(x1),
         static_cast<int>(y1),
         static_cast<int>(z1)
     };
    // Count how many messages we received (including our own)
    // This is all those which fall within the 3x3x3 Moore neighbourhood
    // Not our search radius
    for (const auto &message : FLAMEGPU->message_in(x1, y1, z1)) {
         int messageBin[3] = {
             static_cast<int>(message.getVariable<float>("x")),
             static_cast<int>(message.getVariable<float>("y")),
             static_cast<int>(message.getVariable<float>("z"))
        };
        bool isBad = false;
        for (unsigned int i = 0; i < 3; ++i) {  // Iterate axis
            int binDiff = myBin[i] - messageBin[i];
            if (binDiff > 1 || binDiff < -1) {
                isBad = true;
            }
        }
        count++;
        badCount = isBad ? badCount + 1 : badCount;
    }
    FLAMEGPU->setVariable<unsigned int>("count", count);
    FLAMEGPU->setVariable<unsigned int>("badCount", badCount);
    return flamegpu::ALIVE;
}
"""

count3D = """
FLAMEGPU_AGENT_FUNCTION(count3D, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    unsigned int count = 0;
    // Count how many messages we received (including our own)
    // This is all those which fall within the 3x3x3 Moore neighbourhood
    for (const auto &message : FLAMEGPU->message_in(0, 0, 0)) {
        count++;
    }
    FLAMEGPU->setVariable<unsigned int>("count", count);
    return flamegpu::ALIVE;
}
"""

class Spatial3DMessageTest(TestCase):


    def test_Mandatory(self): 
        bin_counts = {}
        # Construct model
        model = pyflamegpu.ModelDescription("Spatial3DMessageTestModel")
        # Location message
        message = model.newMessageSpatial3D("location")
        message.setMin(0, 0, 0)
        message.setMax(5, 5, 5)
        message.setRadius(1)
        # 5x5x5 bins, total 125
        message.newVariableInt("id")  # unused by current test

        # Circle agent
        agent = model.newAgent("agent")
        agent.newVariableInt("id")
        agent.newVariableFloat("x")
        agent.newVariableFloat("y")
        agent.newVariableFloat("z")
        agent.newVariableUInt("myBin")  # This will be presumed bin index of the agent, might not use this
        agent.newVariableUInt("count")  # Store the distance moved here, for validation
        agent.newVariableUInt("badCount")  # Store how many messages are out of range
        of = agent.newRTCFunction("out", out_mandatory3D)
        of.setMessageOutput("location")
        inf = agent.newRTCFunction("in", in3D)
        inf.setMessageInput("location")

        # Layer #1
        layer = model.newLayer()
        layer.addAgentFunction(of)

        # Layer #2
        layer = model.newLayer()
        layer.addAgentFunction(inf)

        cudaSimulation = pyflamegpu.CUDASimulation(model)

        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents (TODO)
        
        # Currently population has not been init, so generate an agent population on the fly
        for i in range(AGENT_COUNT): 
            instance = population[i]
            instance.setVariableInt("id", i)
            pos =  [rand.uniform(0.0, 5.0), rand.uniform(0.0, 5.0), rand.uniform(0.0, 5.0)] 
            instance.setVariableFloat("x", pos[0])
            instance.setVariableFloat("y", pos[1])
            instance.setVariableFloat("z", pos[2])
            # Solve the bin index
            bin_pos = [int(pos[0]), int(pos[1]), int(pos[2])]
            bin_index = bin_pos[2] * 5 * 5 + bin_pos[1] * 5 + bin_pos[0]
            instance.setVariableUInt("myBin", bin_index)
            # Create it if it doesn't already exist
            if not bin_index in bin_counts: 
                bin_counts[bin_index] = 0
            # increment bin count
            bin_counts[bin_index] += 1
        
        cudaSimulation.setPopulationData(population)


        # Generate results expectation
        bin_results = {}
        # Iterate host bin
        for x1 in range(5):
            for y1 in range(5):
                for z1 in range(5):
                    # Solve the bin index
                    bin_pos1 = [x1, y1, z1]
                    bin_index1 = bin_pos1[2] * 5 * 5 + bin_pos1[1] * 5 + bin_pos1[0]
                    # Count our neighbours
                    count_sum = 0
                    for x2 in range(-1, 2):
                        bin_pos2 = [bin_pos1[0] + x2, 0, 0]
                        for y2 in range(-1, 2): 
                            bin_pos2[1] = bin_pos1[1] + y2
                            for z2 in range(-1, 2): 
                                bin_pos2[2] = bin_pos1[2] + z2
                                # Ensure bin is in bounds
                                if (bin_pos2[0] >= 0 and
                                    bin_pos2[1] >= 0 and
                                    bin_pos2[2] >= 0 and
                                    bin_pos2[0] < 5 and
                                    bin_pos2[1] < 5 and
                                    bin_pos2[2] < 5): 
                                    bin_index2 = bin_pos2[2] * 5 * 5 + bin_pos2[1] * 5 + bin_pos2[0]
                                    count_sum += bin_counts[bin_index2]
                            
                    bin_results[bin_index1] = count_sum

        # Execute a single step of the model
        cudaSimulation.step()

        # Recover the results and check they match what was expected

        cudaSimulation.getPopulationData(population)
        # Validate each agent has same result
        badCountWrong = 0
        for ai in population:
            myBin = ai.getVariableUInt("myBin")
            myResult = ai.getVariableUInt("count")
            assert myResult == bin_results[myBin]
            if ai.getVariableUInt("badCount"):
                badCountWrong += 1
        assert badCountWrong == 0


    def test_Optional(self): 
        bin_counts = {}
        bin_counts_optional = {}
        # Construct model
        model = pyflamegpu.ModelDescription("Spatial3DMessageTestModel")
        # Location message
        message = model.newMessageSpatial3D("location")
        message.setMin(0, 0, 0)
        message.setMax(5, 5, 5)
        message.setRadius(1)
        # 5x5x5 bins, total 125
        message.newVariableInt("id")  # unused by current test

        # Circle agent
        agent = model.newAgent("agent")
        agent.newVariableInt("id")
        agent.newVariableFloat("x")
        agent.newVariableFloat("y")
        agent.newVariableFloat("z")
        agent.newVariableInt("do_output")
        agent.newVariableUInt("myBin")  # This will be presumed bin index of the agent, might not use this
        agent.newVariableUInt("count")  # Store the distance moved here, for validation
        agent.newVariableUInt("badCount")  # Store how many messages are out of range
        of = agent.newRTCFunction("out", out_optional3D)
        of.setMessageOutput("location")
        of.setMessageOutputOptional(True)
        inf = agent.newRTCFunction("in", in3D)
        inf.setMessageInput("location")

        # Layer #1
        layer = model.newLayer()
        layer.addAgentFunction(of)

        # Layer #2
        layer = model.newLayer()
        layer.addAgentFunction(inf)

        cudaSimulation = pyflamegpu.CUDASimulation(model)

        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents (TODO)
        
        # Currently population has not been init, so generate an agent population on the fly
        for i in range(AGENT_COUNT): 
            instance = population[i]
            instance.setVariableInt("id", i)
            pos =  [rand.uniform(0.0, 5.0), rand.uniform(0.0, 5.0), rand.uniform(0.0, 5.0)] 
            if rand.uniform(0.0, 5.0) < 4.0: # 80% chance of output
                do_output = 1 
            else: 
                do_output = 0  
            instance.setVariableFloat("x", pos[0])
            instance.setVariableFloat("y", pos[1])
            instance.setVariableFloat("z", pos[2])
            # Solve the bin index
            bin_pos = [int(pos[0]), int(pos[1]), int(pos[2])]
            bin_index = bin_pos[2] * 5 * 5 + bin_pos[1] * 5 + bin_pos[0]
            instance.setVariableUInt("myBin", bin_index)
            instance.setVariableInt("do_output", do_output)
            # Create it if it doesn't already exist
            if not bin_index in bin_counts: 
                bin_counts[bin_index] = 0
            # increment bin count
            bin_counts[bin_index] += 1
            if (do_output) :
                if not bin_index in bin_counts_optional: 
                    bin_counts_optional[bin_index] = 0
                bin_counts_optional[bin_index] += 1
        
        cudaSimulation.setPopulationData(population)


        # Generate results expectation
        bin_results = {}
        bin_results_optional = {}
        # Iterate host bin
        for x1 in range(5):
            for y1 in range(5):
                for z1 in range(5):
                    # Solve the bin index
                    bin_pos1 = [x1, y1, z1]
                    bin_index1 = bin_pos1[2] * 5 * 5 + bin_pos1[1] * 5 + bin_pos1[0]
                    # Count our neighbours
                    count_sum = 0
                    count_sum_optional = 0
                    for x2 in range(-1, 2):
                        bin_pos2 = [bin_pos1[0] + x2, 0, 0]
                        for y2 in range(-1, 2): 
                            bin_pos2[1] = bin_pos1[1] + y2
                            for z2 in range(-1, 2): 
                                bin_pos2[2] = bin_pos1[2] + z2
                                # Ensure bin is in bounds
                                if (bin_pos2[0] >= 0 and
                                    bin_pos2[1] >= 0 and
                                    bin_pos2[2] >= 0 and
                                    bin_pos2[0] < 5 and
                                    bin_pos2[1] < 5 and
                                    bin_pos2[2] < 5): 
                                    bin_index2 = bin_pos2[2] * 5 * 5 + bin_pos2[1] * 5 + bin_pos2[0]
                                    count_sum += bin_counts[bin_index2]
                                    count_sum_optional += bin_counts_optional[bin_index2]
                            
                    bin_results[bin_index1] = count_sum
                    bin_results_optional[bin_index1] = count_sum_optional

        # Execute a single step of the model
        cudaSimulation.step()

        # Recover the results and check they match what was expected

        cudaSimulation.getPopulationData(population)
        # Validate each agent has same result
        badCountWrong = 0
        for ai in population:
            myBin = ai.getVariableUInt("myBin")
            myResult = ai.getVariableUInt("count")
            assert myResult == bin_results_optional[myBin]
            if ai.getVariableUInt("badCount"):
                badCountWrong += 1
        assert badCountWrong == 0




    def test_BadRadius(self): 
        model = pyflamegpu.ModelDescription("Spatial3DMessageTestModel")
        message = model.newMessageSpatial3D("location")
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setRadius(0)
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setRadius(-10)
        assert e.value.type() == "InvalidArgument"

    def test_BadMin(self): 
        model = pyflamegpu.ModelDescription("Spatial3DMessageTestModel")
        message = model.newMessageSpatial3D("location")
        message.setMax(5, 5, 5)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setMin(5, 0, 0)
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setMin(0, 5, 0)
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setMin(0, 0, 5)
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setMin(6, 0, 0)
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setMin(0, 6, 0)
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setMin(0, 0, 6)
        assert e.value.type() == "InvalidArgument"

    def test_BadMax(self): 
        model = pyflamegpu.ModelDescription("Spatial3DMessageTestModel")
        message = model.newMessageSpatial3D("location")
        message.setMin(5, 5, 5)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setMax(5, 0, 0)
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setMax(0, 5, 0)
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setMax(0, 0, 5)
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setMax(6, 0, 0)
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setMax(0, 6, 0)
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setMax(0, 0, 6)
        assert e.value.type() == "InvalidArgument"

    def test_UnsetMax(self): 
        model = pyflamegpu.ModelDescription("Spatial3DMessageTestModel")
        message = model.newMessageSpatial3D("location")
        message.setMin(5, 5, 5)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            m = pyflamegpu.CUDASimulation(model)
        assert e.value.type() == "InvalidMessage"

    def test_UnsetMin(self): 
        model = pyflamegpu.ModelDescription("Spatial3DMessageTestModel")
        message = model.newMessageSpatial3D("location")
        message.setMin(5, 5, 5)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            m = pyflamegpu.CUDASimulation(model)
        assert e.value.type() == "InvalidMessage"

    def test_reserved_name(self): 
        model = pyflamegpu.ModelDescription("Spatial3DMessageTestModel")
        message = model.newMessageSpatial3D("location")
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.newVariableInt("_")
        assert e.value.type() == "ReservedName"

    def test_ReadEmpty(self): 
        # What happens if we read a message list before it has been output?
        model = pyflamegpu.ModelDescription("Model")
        # Location message
        message = model.newMessageSpatial3D("location")
        message.setMin(-3, -3, -3)
        message.setMax(3, 3, 3)
        message.setRadius(2)
        message.newVariableInt("id")  # unused by current test

        # Circle agent
        agent = model.newAgent("agent")
        agent.newVariableUInt("count", 0)  # Count the number of messages read
        fin = agent.newRTCFunction("in", count3D)
        fin.setMessageInput("location")

        # Layer #1
        layer = model.newLayer()
        layer.addAgentFunction(fin)
        
        # Create 1 agent
        pop_in = pyflamegpu.AgentVector(model.Agent("agent"), 1)
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        cudaSimulation.setPopulationData(pop_in)
        # Execute model
        cudaSimulation.step()
        # Check result
        pop_out = pyflamegpu.AgentVector(model.Agent("agent"), 1)
        pop_out[0].setVariableUInt("count", 1)
        cudaSimulation.getPopulationData(pop_out)
        assert len(pop_out) == 1
        ai = pop_out[0]
        assert ai.getVariableUInt("count") == 0
