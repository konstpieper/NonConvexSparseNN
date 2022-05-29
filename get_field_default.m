function value = get_field_default(structure, field, default)

if isfield(structure, field)
    value = getfield(structure, field);
else
    value = default;
end

end

