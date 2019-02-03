
import JSON

function getConfig()
    config = Dict()
    open("config.json","r") do io
        dicttext = read(io, String)
        config = JSON.parse(dicttext)
    end
    return config


# function save_obj(obj, name)
#     stringdata = JSON.json(obj)
#     open(name + ".json", "w") do io
#         write(io, stringdata)
#     end
# end

# def load_obj(name):
#     with open(name + '.json', 'r') as f:
#         return json.load(f)
end
