import pytest
from unittest import TestCase
from pyflamegpu import *
import random as rand



MODEL_NAME = "Model"
AGENT_NAME = "Agent"
MESSAGE_NAME = "Message"
IN_FUNCTION_NAME = "InFunction"
OUT_FUNCTION_NAME = "OutFunction"
IN_LAYER_NAME = "InLayer"
OUT_LAYER_NAME = "OutLayer"
SQRT_AGENT_COUNT = 12 # This is hard coded in the device functions so can not be changed
AGENT_COUNT = SQRT_AGENT_COUNT * (SQRT_AGENT_COUNT+1)
UINT_MAX = 4294967295

OutFunction = """
FLAMEGPU_AGENT_FUNCTION(OutFunction, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("message_write");
    FLAMEGPU->message_out.setVariable<unsigned int>("index_times_3", index * 3);
    const unsigned int index_x = index % 12;
    const unsigned int index_y = index / 12;
    FLAMEGPU->message_out.setIndex(index_x, index_y);
    return flamegpu::ALIVE;
}
"""

OutOptionalFunction = """
FLAMEGPU_AGENT_FUNCTION(OutOptionalFunction, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("message_write");
    if (index % 2 == 0) {
        FLAMEGPU->message_out.setVariable<unsigned int>("index_times_3", index * 3);
        const unsigned int index_x = index % 12;
        const unsigned int index_y = index / 12;
        FLAMEGPU->message_out.setIndex(index_x, index_y);
    }
    return flamegpu::ALIVE;
}
"""

OutBad = """
FLAMEGPU_AGENT_FUNCTION(OutBad, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    unsigned int index = FLAMEGPU->getVariable<unsigned int>("message_write");
    FLAMEGPU->message_out.setVariable<unsigned int>("index_times_3", index * 3);
    index = index == 13 ? 0 : index;
    const unsigned int index_x = index % 12;
    const unsigned int index_y = index / 12;
    FLAMEGPU->message_out.setIndex(index_x, index_y);
    return flamegpu::ALIVE;
}
"""

InFunction = """
FLAMEGPU_AGENT_FUNCTION(InFunction, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    const unsigned int index_x = my_index % 12;
    const unsigned int index_y = my_index / 12;
    const auto &message = FLAMEGPU->message_in.at(index_x, index_y);
    FLAMEGPU->setVariable("message_read", message.getVariable<unsigned int>("index_times_3"));
    return flamegpu::ALIVE;
}
"""

OutSimple = """
FLAMEGPU_AGENT_FUNCTION(OutSimple, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    const unsigned int index_x = index % 12;
    const unsigned int index_y = index / 12;
    FLAMEGPU->message_out.setIndex(index_x, index_y);
    return flamegpu::ALIVE;
}
"""

MooreTest1 = """
FLAMEGPU_AGENT_FUNCTION(MooreTest1, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    const unsigned int index_x = my_index % 12;
    const unsigned int index_y = my_index / 12;

    // Iterate and check it aligns
    auto filter = FLAMEGPU->message_in.wrap(index_x, index_y);
    auto message = filter.begin();
    unsigned int message_read = 0;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            // Skip ourself
            if (!(i == 0 && j == 0)) {
                // Wrap over boundaries
                const unsigned int their_x = (index_x + i + FLAMEGPU->message_in.getDimX()) % FLAMEGPU->message_in.getDimX();
                const unsigned int their_y = (index_y + j + FLAMEGPU->message_in.getDimY()) % FLAMEGPU->message_in.getDimY();
                if (message->getX() == their_x && message->getY() == their_y)
                    message_read++;
                ++message;
            }
        }
    }
    FLAMEGPU->setVariable<unsigned int>("message_read", message_read);
    return flamegpu::ALIVE;
}
"""

MooreTest2 = """
FLAMEGPU_AGENT_FUNCTION(MooreTest2, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    const unsigned int index_x = my_index % 12;
    const unsigned int index_y = my_index / 12;

    // Iterate and check it aligns
    auto filter = FLAMEGPU->message_in.wrap(index_x, index_y, 2);
    auto message = filter.begin();
    unsigned int message_read = 0;
    for (int i = -2; i <= 2; ++i) {
        for (int j = -2; j <= 2; ++j) {
            // Skip ourself
            if (!(i == 0 && j == 0)) {
                // Wrap over boundaries
                const unsigned int their_x = (index_x + i + FLAMEGPU->message_in.getDimX()) % FLAMEGPU->message_in.getDimX();
                const unsigned int their_y = (index_y + j + FLAMEGPU->message_in.getDimY()) % FLAMEGPU->message_in.getDimY();
                if (message->getX() == their_x && message->getY() == their_y)
                    message_read++;
                ++message;
            }
        }
    }
    FLAMEGPU->setVariable<unsigned int>("message_read", message_read);
    return flamegpu::ALIVE;
}
"""

countArray2D = """
FLAMEGPU_AGENT_FUNCTION(countArray2D, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    unsigned int value = FLAMEGPU->message_in.at(0, 0).getVariable<unsigned int>("value");
    FLAMEGPU->setVariable<unsigned int>("value", value);
    return flamegpu::ALIVE;
}
"""

class TestMessage_Array2D(TestCase):

    def test_Mandatory(self): 
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        message = m.newMessageArray2D(MESSAGE_NAME)
        message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1)
        message.newVariableUInt("index_times_3")
        a = m.newAgent(AGENT_NAME)
        a.newVariableUInt("index")
        a.newVariableUInt("message_read", UINT_MAX)
        a.newVariableUInt("message_write")
        fo = a.newRTCFunction(OUT_FUNCTION_NAME, OutFunction)
        fo.setMessageOutput(message)
        fi = a.newRTCFunction(IN_FUNCTION_NAME, InFunction)
        fi.setMessageInput(message)
        lo = m.newLayer(OUT_LAYER_NAME)
        lo.addAgentFunction(fo)
        li = m.newLayer(IN_LAYER_NAME)
        li.addAgentFunction(fi)
        # Create a list of numbers
        numbers = [0] * AGENT_COUNT
        for i in range(AGENT_COUNT):
            numbers[i] = i
        
        # Shuffle the list of numbers
        rand.shuffle(numbers)
        # Assign the numbers in shuffled order to agents
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            ai = pop[i]
            ai.setVariableUInt("index", i)
            ai.setVariableUInt("message_read", UINT_MAX)
            ai.setVariableUInt("message_write", numbers[i])
        
        # Set pop in model
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)
        c.step()
        c.getPopulationData(pop)
        # Validate each agent has same result
        for ai in pop:
            index = ai.getVariableUInt("index")
            message_read = ai.getVariableUInt("message_read")
            assert index * 3 == message_read
        

    def test_Optional(self): 
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        message = m.newMessageArray2D(MESSAGE_NAME)
        message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1)
        message.newVariableUInt("index_times_3")
        a = m.newAgent(AGENT_NAME)
        a.newVariableUInt("index")
        a.newVariableUInt("message_read", UINT_MAX)
        a.newVariableUInt("message_write")
        fo = a.newRTCFunction(OUT_FUNCTION_NAME, OutOptionalFunction)
        fo.setMessageOutput(message)
        fo.setMessageOutputOptional(True)
        fi = a.newRTCFunction(IN_FUNCTION_NAME, InFunction)
        fi.setMessageInput(message)
        lo = m.newLayer(OUT_LAYER_NAME)
        lo.addAgentFunction(fo)
        li = m.newLayer(IN_LAYER_NAME)
        li.addAgentFunction(fi)
        # Create a list of numbers
        numbers = [0] * AGENT_COUNT
        for i in range(AGENT_COUNT):
            numbers[i] = i
        # Shuffle the list of numbers
        rand.shuffle(numbers)
        # Assign the numbers in shuffled order to agents
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            ai = pop[i]
            ai.setVariableUInt("index", i)
            ai.setVariableUInt("message_read", UINT_MAX)
            ai.setVariableUInt("message_write", numbers[i])
        
        # Set pop in model
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)
        c.step()
        c.getPopulationData(pop)
        # Validate each agent has same result
        for ai in pop:
            index = ai.getVariableUInt("index")
            message_read = ai.getVariableUInt("message_read")
            # index = index % 2 == 0 ? index : 0
            if index % 2 == 0:
                index = index
            else:
                index = 0
            assert index * 3 == message_read
        

    def test_Moore1W(self): 
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        message = m.newMessageArray2D(MESSAGE_NAME)
        message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1)
        a = m.newAgent(AGENT_NAME)
        a.newVariableUInt("index")
        a.newVariableUInt("message_read", UINT_MAX)
        fo = a.newRTCFunction(OUT_FUNCTION_NAME, OutSimple)
        fo.setMessageOutput(message)
        fi = a.newRTCFunction(IN_FUNCTION_NAME, MooreTest1)
        fi.setMessageInput(message)
        lo = m.newLayer(OUT_LAYER_NAME)
        lo.addAgentFunction(fo)
        li = m.newLayer(IN_LAYER_NAME)
        li.addAgentFunction(fi)
        # Assign the numbers in shuffled order to agents
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            ai = pop[i]
            ai.setVariableUInt("index", i)
            ai.setVariableUInt("message_read", UINT_MAX)
        
        # Set pop in model
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)
        c.step()
        c.getPopulationData(pop)
        # Validate each agent has read 8 correct messages
        for ai in pop:
            message_read = ai.getVariableUInt("message_read")
            assert 8 == message_read
        

    def test_Moore2W(self): 
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        message = m.newMessageArray2D(MESSAGE_NAME)
        message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1)
        a = m.newAgent(AGENT_NAME)
        a.newVariableUInt("index")
        a.newVariableUInt("message_read", UINT_MAX)
        fo = a.newRTCFunction(OUT_FUNCTION_NAME, OutSimple)
        fo.setMessageOutput(message)
        fi = a.newRTCFunction(IN_FUNCTION_NAME, MooreTest2)
        fi.setMessageInput(message)
        lo = m.newLayer(OUT_LAYER_NAME)
        lo.addAgentFunction(fo)
        li = m.newLayer(IN_LAYER_NAME)
        li.addAgentFunction(fi)
        # Assign the numbers in shuffled order to agents
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            ai = pop[i]
            ai.setVariableUInt("index", i)
            ai.setVariableUInt("message_read", UINT_MAX)
        
        # Set pop in model
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)
        c.step()
        c.getPopulationData(pop)
        # Validate each agent has read 8 correct messages
        for ai in pop:
            message_read = ai.getVariableUInt("message_read")
            assert 24 == message_read
        

    # Exception tests
    def test_DuplicateOutputException(self): 
        if not pyflamegpu.SEATBELTS:
            pytest.skip("Test requires SEATBELTS to be ON")
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        message = m.newMessageArray2D(MESSAGE_NAME)
        message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1)
        message.newVariableUInt("index_times_3")
        a = m.newAgent(AGENT_NAME)
        a.newVariableUInt("index")
        a.newVariableUInt("message_read", UINT_MAX)
        a.newVariableUInt("message_write")
        fo = a.newRTCFunction(OUT_FUNCTION_NAME, OutBad)
        fo.setMessageOutput(message)
        fi = a.newRTCFunction(IN_FUNCTION_NAME, InFunction)
        fi.setMessageInput(message)
        lo = m.newLayer(OUT_LAYER_NAME)
        lo.addAgentFunction(fo)
        li = m.newLayer(IN_LAYER_NAME)
        li.addAgentFunction(fi)
        # Create a list of numbers
        numbers = [0] * AGENT_COUNT
        for i in range(AGENT_COUNT):
            numbers[i] = i
        
        # Shuffle the list of numbers
        rand.shuffle(numbers)
        # Assign the numbers in shuffled order to agents
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            ai = pop[i]
            ai.setVariableUInt("index", i)
            ai.setVariableUInt("message_read", UINT_MAX)
            ai.setVariableUInt("message_write", numbers[i])
        
        # Set pop in model
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            c.step()
        assert e.value.type() == "ArrayMessageWriteConflict"

    def test_ArrayLenZeroException(self): 
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        message = m.newMessageArray2D(MESSAGE_NAME)       
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setDimensions(0, SQRT_AGENT_COUNT)
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setDimensions([ 0, SQRT_AGENT_COUNT ])
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setDimensions(SQRT_AGENT_COUNT, 0)
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setDimensions([ SQRT_AGENT_COUNT, 0 ])
        assert e.value.type() == "InvalidArgument"

    def test_UnsetLength(self): 
        model = pyflamegpu.ModelDescription(MODEL_NAME)
        model.newMessageArray2D(MESSAGE_NAME)
        # message.setLength(5)  # Intentionally commented out
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            m = pyflamegpu.CUDASimulation(model)
        assert e.value.type() == "InvalidMessage"

    def test_reserved_name(self): 
        model = pyflamegpu.ModelDescription(MODEL_NAME)
        message = model.newMessageArray2D(MESSAGE_NAME)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.newVariableInt("_")
        assert e.value.type() == "ReservedName"

    def test_ReadEmpty(self): 
        # What happens if we read a message list before it has been output?
        model = pyflamegpu.ModelDescription(MODEL_NAME)
        # Location message
        message = model.newMessageArray2D("location")
        message.setDimensions(2,2)
        message.newVariableInt("id")  # unused by current test
        message.newVariableUInt("value")
        # Circle agent
        agent = model.newAgent("agent")
        agent.newVariableUInt("value", 32323)  # Count the number of messages read
        cf = agent.newRTCFunction("in", countArray2D)
        cf.setMessageInput("location")
        # Layer #1
        layer = model.newLayer()
        layer.addAgentFunction(cf)
        # Create 1 agent
        pop_in = pyflamegpu.AgentVector(model.Agent("agent"), 1)
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        cudaSimulation.setPopulationData(pop_in)
        # Execute model
        cudaSimulation.step()
        # Check result
        pop_out = pyflamegpu.AgentVector(model.Agent("agent"), 1)
        pop_out.front().setVariableUInt("value", 22221)
        cudaSimulation.getPopulationData(pop_out)
        assert len(pop_out) == 1
        ai = pop_out.front()
        assert ai.getVariableUInt("value") == 0  # Unset array messages should be 0

