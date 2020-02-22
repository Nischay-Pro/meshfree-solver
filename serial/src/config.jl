import JSON
function getConfig()
    open("../config.json","r") do io
        dicttext = read(io, String)
        config = JSON.parse(dicttext)
        close(io)
        return config
    end
end