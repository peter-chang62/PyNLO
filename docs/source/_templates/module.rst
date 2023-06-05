{{ fullname | escape | underline}}
.. automodule:: {{ fullname }}


{%- block modules %}
{%- if modules %}

{{ _('Modules') | underline(line='-')}}
.. autosummary::
   :template: module.rst
   :toctree:
   :caption: Modules
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{%- endif %}
{%- endblock %}


{%- block attributes %}
{%- if attributes %}

{{ _('Module Attributes') | underline(line='-')}}
.. autosummary::
   :template: base.rst
   :toctree:
   :caption: Module Attributes
{% for item in attributes %}
   {{ item }}
{%- endfor %}
{%- endif %}
{%- endblock %}


{%- block functions %}
{%- if functions %}

{{ _('Functions') | underline(line='-') }}
.. autosummary::
   :template: base.rst
   :toctree:
   :caption: Functions
{% for item in functions %}
   {{ item }}
{%- endfor %}
{%- endif %}
{%- endblock %}


{%- block classes %}
{%- if classes %}

{{ _('Classes')  | underline(line='-')}}
.. autosummary::
   :template: class.rst
   :toctree:
   :caption: Classes
{% for item in classes %}
   {{ item }}
{%- endfor %}
{%- endif %}
{%- endblock %}


{%- block exceptions %}
{%- if exceptions %}

{{ _('Exceptions') | underline(line='-') }}
.. autosummary::
   :template: base.rst
   :toctree:
   :caption: Exceptions
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{%- endif %}
{%- endblock %}
