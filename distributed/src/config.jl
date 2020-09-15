import JSON
function getConfig()
    open("../config.json","r") do io
        dicttext = read(io, String)
        config = JSON.parse(dicttext)
        close(io)
        return config
    end
end

# function save_obj(obj, name)
#     stringdata = JSON.json(obj)
#     open(name + ".json", "w") do io
#         write(io, stringdata)
#     end
# end

# def load_obj(name):
#     with open(name + '.json', 'r') as f:
#         return json.load(f)
